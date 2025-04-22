# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Extension to Elevenlabs services for improved ACE compatability."""

import base64
import json
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs import ElevenLabsTTSService, calculate_word_times

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


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
        async for frame in super().run_tts(text):
            yield frame

    async def _receive_messages(self):
        """Processes incoming websocket messages.

        Handles audio data and alignment information, emitting appropriate frames.
        """
        async for message in self._get_websocket():
            msg = json.loads(message)

            is_boundary_marker_in_alignment = False
            is_skip_message = False
            if msg.get("alignment"):
                # Check if the boundary marker character is in the alignment
                chars = msg.get("alignment").get("chars")
                logger.debug(f"received alignment chars: {chars[0:3]}")
                if self._boundary_marker_character in chars:
                    is_boundary_marker_in_alignment = True
                    # If the boundary marker is the first character, it is safe to
                    # not send the associated audio downstream.
                    # This helps preventing audio glitches.
                    if self._boundary_marker_character == chars[0]:
                        is_skip_message = True

            if msg.get("audio") and not is_skip_message:
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()

                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                await self.push_frame(frame)

            if msg.get("alignment") and not is_skip_message:
                msg["alignment"] = self._shift_partial_words(msg["alignment"])
                word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                await self.add_word_timestamps(word_times)
                self._cumulative_time = word_times[-1][1]

            if is_boundary_marker_in_alignment:
                await self.push_frame(TTSStoppedFrame())

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
