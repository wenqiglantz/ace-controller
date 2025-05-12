# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

"""ElevenLabs services implementation for ACE pipelines.

This module provides integration with ElevenLabs' speech services, including:
- Text-to-Speech (TTS) with end-of-speech detection for better avatar synchronization
- Automatic Speech Recognition (ASR) with Scribe v1 model
- Support for multiple languages, speaker diarization, and word-level timestamps

For documentation on ElevenLabs Speech services, please refer to:
https://elevenlabs.io/docs/
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from typing import Any, Dict

import aiohttp
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.services.elevenlabs import ElevenLabsTTSService, calculate_word_times
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced
from pipecat.transcriptions.language import Language


@traceable
class ElevenLabsTTSServiceWithEndOfSpeech(ElevenLabsTTSService):
    """ElevenLabs TTS service with end-of-speech detection.

    This class extends the base ElevenLabs TTS service to add functionality for detecting
    and handling the end of speech segments. It uses a special character to identify speech boundaries
    to send out TTSStoppedFrames at the right times. This is useful for interactive avatar experiences
    where TTSStoppedFrames are required to signal the end of a speech segment to control lip movement
    of the avatar.

    Attributes:
        boundary_marker_character: Character marking speech boundaries.

    Input frames:
        TextFrame: Text to synthesize into speech.
        TTSSpeakFrame: Alternative text input for speech synthesis.
        LLMFullResponseEndFrame: Signals LLM response completion.
        BotStoppedSpeakingFrame: Signals bot speech completion.

    Output frames:
        TTSStartedFrame: Signals TTS start.
        TTSTextFrame: Contains text being synthesized.
        TTSAudioRawFrame: Contains raw audio data.
        TTSStoppedFrame: Signals TTS completion.
    """

    def __init__(self, boundary_marker_character: str = "\u200b", *args, **kwargs):
        """Initialize the ElevenLabsTTSServiceWithEndOfSpeech.

        Shares all the parameters with the parent class ElevenLabsTTSService but adds
        the boundary_marker_character parameter to identify speech boundaries.

        Args:
            boundary_marker_character (str): Character used to mark speech boundaries.
                Defaults to zero-width space. Should be a character that is not used in the text
                Ideally the character is not printable (to avoid showing the character in transcripts)
            *args: Variable length argument list passed to parent ElevenLabsTTSService.
            **kwargs: Arbitrary keyword arguments passed to parent ElevenLabsTTSService.
        """
        super().__init__(*args, **kwargs)
        self._boundary_marker_character = boundary_marker_character
        self._partial_word: dict | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes frames.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._partial_word = None

    async def flush_audio(self):
        """Flushes remaining audio in websocket connection.

        Sends special marker messages to flush audio buffer and signal end of speech.
        """
        if self._websocket:
            logger.debug("11labs: Flushing audio")
            msg = {"text": f"{self._boundary_marker_character} "}
            await self._websocket.send(json.dumps(msg))
            msg = {"text": " ", "flush": True}
            await self._websocket.send(json.dumps(msg))

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="tts")
    async def run_tts(self, text: str):
        """Run text-to-speech synthesis.

        Compared to the based class method this method is instrumented for tracing.
        """
        logger.info(f"Starting TTS for text: [{text}]")
        try:
            async for frame in super().run_tts(text):
                if isinstance(frame, TTSAudioRawFrame):
                    logger.info(f"Received TTS audio frame of size: {len(frame.audio_data)} bytes")
                elif isinstance(frame, TTSStoppedFrame):
                    logger.info("Received TTS stopped frame")
                yield frame
        except Exception as e:
            logger.error(f"Error in TTS service: {e}")
            # Try to yield a dummy audio frame to ensure audio flow
            yield TTSAudioRawFrame(b'\x00' * 1000, self.sample_rate, 1)
            yield TTSStoppedFrame()

    async def _receive_messages(self):
        """Processes incoming websocket messages.

        Handles audio data and alignment information, emitting appropriate frames.
        """
        logger.info("Starting to receive TTS messages from websocket")
        try:
            async for message in self._get_websocket():
                logger.debug(f"Received websocket message of length: {len(message)}")
                msg = json.loads(message)

                is_boundary_marker_in_alignment = False
                is_skip_message = False
                if msg.get("alignment"):
                    # Check if the boundary marker character is in the alignment
                    chars = msg.get("alignment").get("chars")
                    logger.debug(f"Received alignment chars: {chars}")
                    if self._boundary_marker_character in chars:
                        logger.info("Boundary marker found in alignment")
                        is_boundary_marker_in_alignment = True
                        # If the boundary marker is the first character, it is safe to
                        # not send the associated audio downstream.
                        # This helps preventing audio glitches.
                        if self._boundary_marker_character == chars[0]:
                            logger.info("Skipping message because boundary marker is first character")
                            is_skip_message = True

                if msg.get("audio") and not is_skip_message:
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()

                    audio = base64.b64decode(msg["audio"])
                    logger.info(f"Sending audio of length {len(audio)} bytes downstream")
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)

                if msg.get("alignment") and not is_skip_message:
                    msg["alignment"] = self._shift_partial_words(msg["alignment"])
                    word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                    await self.add_word_timestamps(word_times)
                    self._cumulative_time = word_times[-1][1]

                if is_boundary_marker_in_alignment:
                    logger.info("Sending TTSStoppedFrame")
                    await self.push_frame(TTSStoppedFrame())
        except Exception as e:
            logger.error(f"Error in _receive_messages: {e}")

    def _shift_partial_words(self, alignment_info: dict[str, Any]) -> dict[str, Any]:
        """Shifts partial words from the previous alignment and retains incomplete words."""
        keys = ["chars", "charStartTimesMs", "charDurationsMs"]
        # Add partial word from the previous part
        if self._partial_word:
            for key in keys:
                alignment_info[key] = self._partial_word[key] + alignment_info[key]
            self._partial_word = None

        # Check if the last word is incomplete
        if not alignment_info["chars"][-1].isspace():
            # Find the last space character
            last_space_index = 0
            for i in range(len(alignment_info["chars"]) - 1, -1, -1):
                if alignment_info["chars"][i].isspace():
                    last_space_index = i + 1
                    break

            # Split into completed and partial parts
            self._partial_word = {key: alignment_info[key][last_space_index:] for key in keys}
            for key in keys:
                alignment_info[key] = alignment_info[key][:last_space_index]

        return alignment_info


@traceable
class ElevenLabsASRService(STTService):
    """ElevenLabs Automatic Speech Recognition service.

    Provides speech recognition using ElevenLabs Scribe v1 model with support for:
        - Real-time transcription (with streaming buffer approach)
        - Word-level timestamps
        - Multiple languages
        - Speaker diarization
        - Audio events tagging
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "scribe_v1",
        language: Language | None = Language.EN_US,
        sample_rate: int = 16000,
        max_alternatives: int = 1,
        interim_results: bool = True,
        generate_interruptions: bool = False,  # Only set to True if transport VAD is disabled
        diarize: bool = False,
        num_speakers: int | None = None,
        tag_audio_events: bool = True,
        timestamps_granularity: str = "word",
        chunk_size_seconds: int = 3,  # Buffer this many seconds before sending to API
        idle_timeout: int = 30,  # Timeout for idle ASR request
        file_format: str = "pcm_s16le_16",  # Using the standard PCM format for audio
        **kwargs,
    ):
        """Initialize the ElevenLabs ASR service.

        Args:
            api_key: ElevenLabs API key for accessing the service.
            model: ASR model name (e.g., 'scribe_v1' or 'scribe_v1_experimental').
            language: Language for recognition (optional, auto-detected if not specified).
            sample_rate: Audio sample rate in Hz (16000 is recommended).
            max_alternatives: Maximum number of alternatives (currently only 1 is supported).
            interim_results: Enable interim results for streaming-like experience.
            generate_interruptions: Enable interruption events based on speech detection.
            diarize: Whether to identify different speakers in the audio.
            num_speakers: Maximum number of speakers to identify (1-32).
            tag_audio_events: Whether to tag non-speech audio events in transcription.
            timestamps_granularity: Level of timestamp detail ('word' or 'character').
            chunk_size_seconds: How many seconds of audio to buffer before sending to API.
            idle_timeout: Timeout for idle ASR request in seconds.
            file_format: Format of the audio file ('pcm_s16le_16' or 'other').
            **kwargs: Additional arguments for STTService.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._model = model
        self._language_code = language.name.lower() if language else None
        self._sample_rate = sample_rate
        self._max_alternatives = max_alternatives
        self._interim_results = interim_results
        self._diarize = diarize
        self._num_speakers = num_speakers
        self._tag_audio_events = tag_audio_events
        self._timestamps_granularity = timestamps_granularity
        self._chunk_size_seconds = chunk_size_seconds
        self._idle_timeout = idle_timeout
        self._file_format = file_format
        self.last_transcript_frame = None
        self.set_model_name(model)

        # Initialize audio buffer
        self._audio_buffer = bytearray()
        self._buffer_size_bytes = int(self._sample_rate * 2 * self._chunk_size_seconds)  # 16-bit audio = 2 bytes per sample
        
        self._queue = asyncio.Queue()
        self._generate_interruptions = generate_interruptions
        if self._generate_interruptions:
            self._vad_state = VADState.QUIET

        # Initialize the thread task and response task
        self._thread_task = None
        self._response_task = None
        self._thread_running = False
        self._api_url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        # Keep track of all audio chunk processing tasks
        self._audio_chunk_tasks = set()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: False as this service does not support metric generation.
        """
        return False

    async def start(self, frame: StartFrame):
        """Start the ASR service.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)
        self._response_task = self.create_task(self._response_task_handler())
        self._response_queue = asyncio.Queue()
        self._audio_buffer = bytearray()
        self._thread_running = True

    async def stop(self, frame: EndFrame):
        """Stop the ASR service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        
        # Process any remaining audio in the buffer
        if len(self._audio_buffer) > 0:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()
            # Make sure to track this task like other chunk processing tasks
            task = self.create_task(self._process_audio_chunk(audio_data))
            self._audio_chunk_tasks.add(task)
            task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
        
        # Make sure we stop tasks AFTER processing remaining audio to properly track all tasks
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the ASR service and cleanup resources.

        Args:
            frame: The CancelFrame that triggered the cancellation.
        """
        await super().cancel(frame)
        
        # Process any remaining audio in the buffer
        if len(self._audio_buffer) > 0:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()
            # Make sure to track this task like other chunk processing tasks
            task = self.create_task(self._process_audio_chunk(audio_data))
            self._audio_chunk_tasks.add(task)
            task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
        
        # Make sure we stop tasks AFTER processing remaining audio to properly track all tasks
        await self._stop_tasks()

    async def _stop_tasks(self):
        """Stop all running tasks."""
        self._thread_running = False
        
        # Wait for tasks to be added to the set from any in-progress operations
        await asyncio.sleep(0.1)
        
        if self._thread_task is not None and not self._thread_task.done():
            await self.cancel_task(self._thread_task)
        
        if self._response_task is not None and not self._response_task.done():
            await self.cancel_task(self._response_task)
            
        # Cancel all audio chunk processing tasks
        logger.info(f"Cancelling {len(self._audio_chunk_tasks)} audio chunk processing tasks")
        for task in self._audio_chunk_tasks.copy():
            if not task.done():
                try:
                    await self.cancel_task(task)
                except Exception as e:
                    logger.warning(f"Error cancelling audio chunk task: {e}")
        
        # Give canceled tasks a moment to complete their cancellation
        await asyncio.sleep(0.1)
        self._audio_chunk_tasks.clear()

    def create_task(self, coro):
        """Create and track a task.

        Overrides the base create_task method to ensure all tasks are properly tracked.

        Args:
            coro: Coroutine to create a task for.

        Returns:
            The created task.
        """
        task = super().create_task(coro)
        
        # If this is a _process_audio_chunk task, add it to our tracking set
        # Check the coroutine's qualname to identify it
        if coro.__qualname__.endswith('_process_audio_chunk'):
            self._audio_chunk_tasks.add(task)
            task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
            
        return task

    async def _process_audio_chunk(self, audio_data: bytes):
        """Process an audio chunk by sending it to the ElevenLabs API.
        
        Args:
            audio_data: Audio bytes to process.
        """
        try:
            # Check if task is already cancelled to exit early
            if asyncio.current_task().cancelled():
                logger.debug("Audio chunk processing task was cancelled, exiting early")
                return
                
            logger.debug(f"Sending audio chunk of {len(audio_data)} bytes to ElevenLabs ASR API")
            
            # Prepare the data for the API request
            headers = {"xi-api-key": self._api_key}
            
            data = {
                "model_id": self._model,
                "file_format": self._file_format,
                "timestamps_granularity": self._timestamps_granularity,
                "diarize": str(self._diarize).lower(),
                "tag_audio_events": str(self._tag_audio_events).lower(),
            }
            
            if self._language_code:
                data["language_code"] = self._language_code
            
            if self._num_speakers:
                data["num_speakers"] = str(self._num_speakers)
            
            # Make the API request using aiohttp's FormData for file upload
            form_data = aiohttp.FormData()
            
            # Add all the regular data fields
            for key, value in data.items():
                form_data.add_field(key, value)
            
            # Add the file field
            form_data.add_field('file', 
                               audio_data, 
                               filename='audio.raw',
                               content_type='audio/raw')
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                # Set a reasonable timeout for the request
                timeout = aiohttp.ClientTimeout(total=30)
                
                # Check again for cancellation before making request
                if asyncio.current_task().cancelled():
                    logger.debug("Audio chunk processing task was cancelled before API request, exiting")
                    return
                    
                async with session.post(self._api_url, headers=headers, data=form_data, timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        await self._response_queue.put(result)
                    else:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs ASR API error: {response.status} - {error_text}")
            
        except asyncio.CancelledError:
            # Handle task cancellation gracefully
            logger.debug("Audio chunk processing task was cancelled during execution")
            raise
        except Exception as e:
            logger.error(f"Error in ElevenLabs ASR processing: {e}")

    async def _handle_interruptions(self, frame: Frame):
        """Handle interruption events.
        
        Args:
            frame: The user speaking frame that triggered the interruption.
        """
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band.
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

        await self.push_frame(frame)

    async def _handle_response(self, response: Dict[str, Any]):
        """Process ASR response and generate appropriate transcription frames.
        
        Args:
            response: The JSON response from the ElevenLabs API.
        """
        try:
            if "text" not in response or not response["text"]:
                logger.debug("Empty transcript received from ElevenLabs ASR")
                return
            
            transcript = response["text"]
            logger.debug(f"Transcript received from ElevenLabs ASR: [{transcript}]")
            
            # Send the transcript as a final result
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            
            if self._generate_interruptions:
                self._vad_state = VADState.QUIET
                await self._handle_interruptions(UserStoppedSpeakingFrame())
            
            logger.debug(f"Final user transcript: [{transcript}]")
            await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601(), None))
            self.last_transcript_frame = None
            
            # When we're in interim mode, also send partial results based on word timestamps
            if self._interim_results and "words" in response:
                words = response.get("words", [])
                # Process and send interim results if we need to show them in the UI
                # This is primarily for visualization as the main transcript is already sent
                for word_item in words:
                    if word_item.get("type") == "word":
                        # You can send interim frames with individual words if needed
                        pass
            
        except Exception as e:
            logger.error(f"Error handling ElevenLabs ASR response: {e}")

    async def _response_task_handler(self):
        """Handle responses from the ElevenLabs API."""
        while True:
            try:
                response = await self._response_queue.get()
                await self._handle_response(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response task handler: {e}")

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="asr")
    async def _buffer_handler(self):
        """Handle the audio buffer, sending chunks to the API as needed."""
        try:
            self._thread_running = True
            while self._thread_running:
                try:
                    # Wait for the next audio chunk with timeout
                    audio_chunk = await asyncio.wait_for(self._queue.get(), timeout=self._idle_timeout)
                    
                    # Add audio to buffer
                    self._audio_buffer.extend(audio_chunk)
                    
                    # Process buffer if it's large enough
                    if len(self._audio_buffer) >= self._buffer_size_bytes:
                        audio_data = bytes(self._audio_buffer[:self._buffer_size_bytes])
                        self._audio_buffer = self._audio_buffer[self._buffer_size_bytes:]
                        
                        # Create task to process the audio chunk and track it
                        task = self.create_task(self._process_audio_chunk(audio_data))
                        self._audio_chunk_tasks.add(task)
                        task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
                        
                except asyncio.TimeoutError:
                    logger.info(f"ASR service is idle for {self._idle_timeout} seconds")
                    # Process any remaining audio in the buffer before timing out
                    if len(self._audio_buffer) > 0:
                        audio_data = bytes(self._audio_buffer)
                        self._audio_buffer = bytearray()
                        task = self.create_task(self._process_audio_chunk(audio_data))
                        self._audio_chunk_tasks.add(task)
                        task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
                    
                    self._thread_running = False
                    break
                    
        except asyncio.CancelledError:
            self._thread_running = False
            # Process any remaining audio in the buffer before cancelling
            if len(self._audio_buffer) > 0:
                audio_data = bytes(self._audio_buffer)
                self._audio_buffer = bytearray()
                task = self.create_task(self._process_audio_chunk(audio_data))
                self._audio_chunk_tasks.add(task)
                task.add_done_callback(lambda t: self._audio_chunk_tasks.discard(t))
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text recognition.

        Args:
            audio: The audio data to process.

        Yields:
            Frame: A sequence of frames containing the recognition results.
        """
        # If there's a previous thread task, make sure it's done before creating a new one
        if self._thread_task is not None and self._thread_task.done():
            # Check if the task ended with an exception
            try:
                exception = self._thread_task.exception()
                if exception:
                    logger.warning(f"Previous buffer handler task failed with: {exception}")
            except asyncio.InvalidStateError:
                # Task was cancelled, this is expected
                pass
            self._thread_task = None
            
        # Create a new thread task if needed
        if self._thread_task is None:
            logger.info("Starting new buffer handler task")
            self._thread_task = self.create_task(self._buffer_handler())
        
        # In streaming mode, we need to start speech detection as soon as we hear the user
        if self._generate_interruptions and len(audio) > 0 and self._vad_state != VADState.SPEAKING:
            self._vad_state = VADState.SPEAKING
            await self._handle_interruptions(UserStartedSpeakingFrame())
            
        await self._queue.put(audio)
        yield None
