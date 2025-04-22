# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Audio2Face-3D service for processing audio data and sending it to Audio2Face-3D.

You can find more info on the Audio2Face-3D service:
https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html
"""

import asyncio
import time
from asyncio import Queue, Task

import grpc
from grpc.aio import StreamStreamCall
from loguru import logger
from nvidia_ace.audio_pb2 import AudioHeader
from nvidia_audio2face_3d.audio2face_pb2_grpc import A2FControllerServiceStub
from nvidia_audio2face_3d.messages_pb2 import (
    A2F3DAnimationDataStream,
    AudioWithEmotion,
    AudioWithEmotionStream,
    AudioWithEmotionStreamHeader,
)
from opentelemetry import metrics, trace
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pydantic import BaseModel

from nvidia_pipecat.frames.animation import (
    AnimationDataStreamRawFrame,
    AnimationDataStreamStartedFrame,
    AnimationDataStreamStoppedFrame,
)
from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced

# Source ID for Animation Graph mixer
A2F_SOURCE_ID = "Audio2Face with Emotions"


class NimConfig(BaseModel):
    """Contains all the relevant information to connect to an NVCF cloud function deployment of Audio2Face-3D.

    Args:
        api_key: The NGC personal or service key with access to the function_id
        function_id: The function ID of the Audio2Face-3D model to use (defaults to James, see API docs for supported
            models: https://build.nvidia.com/nvidia/audio2face-3d/api)
        target: URL of the NVCF function (defaults to grpc.nvcf.nvidia.com:443)
    """

    api_key: str
    function_id: str = "52f51a79-324c-4dbe-90ad-798ab665ad64"
    target: str = "grpc.nvcf.nvidia.com:443"


@traceable
class Audio2Face3DService(FrameProcessor):
    """Converts streamed audio to facial blendshapes for realtime lipsyncing and facial performances.

    The service connects to the a NVIDIA Audio2Face-3D microservice and sends audio data to it.
    The Audio2Face-3D microservice converts speech into facial animation in the form of ARKit Blendshapes.
    The facial animation includes emotional expression. Where emotions can be detected, the facial
    animation system captures key poses and shapes to replicate character facial performance by
    automatically detecting emotions in the input audio. A rendering engine can consume Blendshape
    topology to display a 3D avatar's performance.

    This service receives facial blendshapes from the Audio2Face-3D microservice and converts them into a Frame stream
    for consumption by the animation graph service.


    Input frames:
        TTSStartedFrame: Indicates start of new TTS audio stream.
        TTSAudioRawFrame: Audio data to be processed.
        TTSStoppedFrame: Indicates end of current TTS audio stream.

    Output frames:
        AnimationDataStreamStartedFrame: Indicates start of new animation stream.
        AnimationDataStreamRawFrame: Audio and facial blendshapes data.
        AnimationDataStreamStoppedFrame: Indicates end of current animation stream.
    """

    def __init__(
        self,
        *,
        target: str = "127.0.0.1:52000",
        sample_rate: int = 16000,
        bits_per_sample: int = 16,
        nim_config: NimConfig | None = None,
        send_silence_on_start: bool = False,
        **kwargs,
    ):
        """Initialize the Audio2Face3DService.

        Args:
            target (str): The URL of the gRPC remote endpoint for Audio2Face-3D. Ignored if a NimConfig is passed.
            sample_rate (int): The sample rate of the audio file. Defaults to 16000.
            bits_per_sample (int): The bits per sample of the audio file. Defaults to 16.
            nim_config (NimConfig, optional): NIM Configuration to connect to an NVCF deployment of Audio2Face-3D.
              Overrides target parameter if used. Defaults to None.
            send_silence_on_start (bool): Whether to send silence frames on startup. Defaults to False.
                Sending silence frames on startup can help to clear the state of any downstream services. (At the moment
                this is e.g. useful to make sure that the animation graph service is reset when the service starts.)
            **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._queue = Queue()
        self._nim_config = nim_config
        if self._nim_config is not None:
            self._channel = grpc.aio.secure_channel(nim_config.target, grpc.ssl_channel_credentials())
        else:
            self._channel = grpc.aio.insecure_channel(target)
        self._stub = A2FControllerServiceStub(self._channel)
        self._sample_rate = sample_rate
        self._bits_per_sample = bits_per_sample
        self._receiving_animation_data_stream_task: Task | None = None
        self._sending_audio_data_task: Task | None = None
        self._a2f_meter = metrics.get_meter("audio2face-3d_processor")
        self._start_timestamp = None
        self._latency_histogram = self._a2f_meter.create_histogram(
            name="a2f-3d.latency",
            description="measures the latency of the first byte coming out of Audio2Face-3D",
            unit="ms",
            explicit_bucket_boundaries_advisory=[
                0,
                5,
                25,
                50,
                75,
                100,
                125,
                150,
                200,
                250,
                300,
                400,
                500,
                750,
                1000,
                2000,
            ],
        )
        self._send_silence_on_start = send_silence_on_start

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="receive animation")
    async def _receive_animation_data_stream(
        self, stream: StreamStreamCall[AudioWithEmotionStream, A2F3DAnimationDataStream]
    ):
        try:
            action_id = "unknown"
            has_started = False
            async for el in stream:
                if el.HasField("animation_data_stream_header"):
                    started_frame = AnimationDataStreamStartedFrame(
                        audio_header=el.animation_data_stream_header.audio_header,
                        animation_header=el.animation_data_stream_header.skel_animation_header,
                        animation_source_id=A2F_SOURCE_ID,
                    )
                    action_id = started_frame.action_id
                    trace.get_current_span().add_event("Received animation data stream header")
                    await self.push_frame(started_frame)
                    has_started = True
                elif el.HasField("animation_data"):
                    if self._start_timestamp:
                        latency = (time.time_ns() - self._start_timestamp) / 1000000
                        self._start_timestamp = None
                        logger.debug(f"Recording latency {latency}")
                        self._latency_histogram.record(latency)
                        trace.get_current_span().add_event("Received first animation packet")
                    await self.push_frame(
                        AnimationDataStreamRawFrame(animation_data=el.animation_data, action_id=action_id)
                    )
            await self.push_frame(AnimationDataStreamStoppedFrame(action_id=action_id))
        except asyncio.CancelledError:
            logger.debug("_receive_animation_data_stream cancelled.")
            raise
        except Exception as e:
            if has_started:
                await self.push_frame(AnimationDataStreamStoppedFrame(action_id=action_id))
            logger.error(f"exception when receiving audio: {e}")

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="send audio")
    async def _sending_audio_data(self):
        logger.debug("_sending_audio_data started")
        stream: StreamStreamCall[AudioWithEmotionStream, A2F3DAnimationDataStream] | None = None
        leftover_byte: bytes = b""
        first = True
        while True:
            try:
                frame = await self._queue.get()
                logger.debug(f"a2f processing frame: {frame} and stream is {stream}")
                if isinstance(frame, TTSStartedFrame):
                    if stream and not stream.done():
                        logger.debug("Got TTSStartedFrame while still processing previous stream")
                        await stream.write(AudioWithEmotionStream(end_of_audio=AudioWithEmotionStream.EndOfAudio()))
                        await stream.done_writing()
                        stream = None
                        await self.wait_for_task(self._receiving_animation_data_stream_task, timeout=0.3)
                        logger.debug("Finished waiting for previous stream to finish")

                    leftover_byte: bytes = b""
                    if self._nim_config is not None:
                        stream = self._stub.ProcessAudioStream(
                            metadata=(
                                ("function-id", self._nim_config.function_id),
                                ("authorization", f"Bearer {self._nim_config.api_key}"),
                            )
                        )
                    else:
                        stream = self._stub.ProcessAudioStream()
                    self._receiving_animation_data_stream_task = self.create_task(
                        self._receive_animation_data_stream(stream)
                    )
                    logger.debug(f"a2f created stream {stream}")
                    await stream.write(
                        AudioWithEmotionStream(
                            audio_stream_header=AudioWithEmotionStreamHeader(
                                audio_header=AudioHeader(
                                    audio_format=AudioHeader.AUDIO_FORMAT_PCM,
                                    channel_count=1,
                                    bits_per_sample=self._bits_per_sample,
                                    samples_per_second=self._sample_rate,
                                )
                            )
                        )
                    )
                    trace.get_current_span().add_event("Audio header sent")

                elif isinstance(frame, TTSAudioRawFrame):
                    # Buffer partial int16 values since Audio2Face-3D requires complete int16 values (int16 is 2 bytes)
                    if leftover_byte:
                        logger.debug("adding leftover byte to frame")
                        frame.audio = leftover_byte + frame.audio
                        leftover_byte = b""

                    # When audio length is uneven
                    if len(frame.audio) % 2 != 0:
                        leftover_byte = frame.audio[-1:]
                        frame.audio = frame.audio[:-1]
                    if stream and not stream.done() and len(frame.audio) > 0:
                        # A2F has the best performance when sending audio in chunks of 1 second of audio
                        # Also we must not exceed the max size which is typically 10s per chunk
                        number_of_bytes_per_sample = 2
                        number_of_chunks = (len(frame.audio) // number_of_bytes_per_sample) // frame.sample_rate + 1
                        logger.debug(f"send audio frame (length: {len(frame.audio)}, num_chunks: {number_of_chunks})")
                        for i in range(number_of_chunks):
                            # Create an audio chunk
                            chunk = frame.audio[
                                i * frame.sample_rate * number_of_bytes_per_sample : i
                                * frame.sample_rate
                                * number_of_bytes_per_sample
                                + frame.sample_rate * number_of_bytes_per_sample
                            ]

                            if len(chunk) > 0:
                                logger.debug(f"sending audio chunk... (length: {len(chunk)})")
                                if first:
                                    self._start_timestamp = time.time_ns()
                                    trace.get_current_span().add_event("First audio frame sent")
                                    first = False
                                await stream.write(
                                    AudioWithEmotionStream(
                                        audio_with_emotion=AudioWithEmotion(audio_buffer=chunk, emotions=[])
                                    )
                                )
                elif isinstance(frame, TTSStoppedFrame):
                    if stream and not stream.done():
                        await stream.write(AudioWithEmotionStream(end_of_audio=AudioWithEmotionStream.EndOfAudio()))
                        await stream.done_writing()
                        stream = None
                        await self.wait_for_task(self._receiving_animation_data_stream_task)
                        logger.debug("TTSStoppedFrame processing completed.")
                        self._start_timestamp = None
                        first = True
                    else:
                        logger.debug("received TTSStoppedFrame with no stream active")
            except asyncio.CancelledError as e:
                logger.debug("_sending_audio_data cancelled.")
                try:
                    if stream and not stream.done():
                        logger.debug("Sending end of audio during cancellation to properly close stream.")
                        await stream.write(AudioWithEmotionStream(end_of_audio=AudioWithEmotionStream.EndOfAudio()))
                        await stream.done_writing()
                except asyncio.CancelledError:
                    raise
                finally:
                    # We need to make sure to raise the cancellation error for proper task cancellation
                    raise e
            except Exception as e:
                stream = None
                self._start_timestamp = None
                first = True
                logger.error(f"exception when sending audio: {e}")

    async def cleanup(self) -> None:
        """Clean up the service."""
        logger.debug("Audio2Face3DService cleanup started")
        await super().cleanup()
        await self._stop_tasks()

    async def _stop_tasks(self) -> None:
        if self._receiving_animation_data_stream_task:
            await self.cancel_task(self._receiving_animation_data_stream_task, timeout=0.3)
            self._receiving_animation_data_stream_task = None
        if self._sending_audio_data_task:
            await self.cancel_task(self._sending_audio_data_task, timeout=0.3)
            self._sending_audio_data_task = None

    async def _clear_queue(self):
        await self._stop_tasks()
        self._queue = asyncio.Queue()
        self._sending_audio_data_task = self.create_task(self._sending_audio_data())

    async def _send_silence(self):
        await self._queue.put(TTSStartedFrame())
        await self._queue.put(
            TTSAudioRawFrame(
                audio=bytes([0] * (self._sample_rate // 10)), sample_rate=self._sample_rate, num_channels=1
            )
        )
        await self._queue.put(TTSStoppedFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and not self._sending_audio_data_task:
            self._sending_audio_data_task = self.create_task(self._sending_audio_data())
            if self._send_silence_on_start:
                await self._send_silence()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._queue.put(frame)
        elif isinstance(frame, TTSStartedFrame | TTSStoppedFrame):
            await self._queue.put(frame)
            await self.push_frame(frame, direction)
        else:
            if isinstance(frame, StartInterruptionFrame):
                logger.debug(f"Interruption recieved: {frame}")
                await self._clear_queue()

            await self.push_frame(frame, direction)
