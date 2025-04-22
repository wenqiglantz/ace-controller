# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Audio2Face 3D Service."""

import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from nvidia_audio2face_3d.messages_pb2 import AudioWithEmotionStream
from pipecat.frames.frames import StartInterruptionFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline

from nvidia_pipecat.frames.animation import (
    AnimationDataStreamRawFrame,
    AnimationDataStreamStartedFrame,
    AnimationDataStreamStoppedFrame,
)
from nvidia_pipecat.services.audio2face_3d_service import Audio2Face3DService
from tests.unit.utils import FrameStorage, ignore, run_interactive_test


class AsyncIterator:
    """Helper class for mocking async iteration in tests.

    Attributes:
        items: List of items to yield during iteration.
    """

    def __init__(self, items):
        """Initialize the async iterator.

        Args:
            items: List of items to yield during iteration.
        """
        self.items = items

    def __aiter__(self):
        """Return self as the iterator.

        Returns:
            AsyncIterator: Self reference for iteration.
        """
        return self

    async def __anext__(self):
        """Get the next item asynchronously.

        Returns:
            Any: Next item in the sequence.

        Raises:
            StopAsyncIteration: When no more items are available.
        """
        try:
            await asyncio.sleep(0.1)
            return self.items.pop(0)
        except IndexError as error:
            raise StopAsyncIteration from error


def get_mock_stream():
    """Create a mock Audio2Face stream for testing.

    Returns:
        MagicMock: A configured mock object that simulates the A2F service stream,
        including header and animation data responses.
    """
    mock_stub = MagicMock(spec=["ProcessAudioStream"])

    # Configure mock responses
    header_response = MagicMock()
    header_response.HasField = lambda x: x == "animation_data_stream_header"
    header_response.animation_data_stream_header = MagicMock(
        audio_header=MagicMock(), skel_animation_header=MagicMock()
    )

    data_response = MagicMock()
    data_response.HasField = lambda x: x == "animation_data"
    data_response.animation_data = "test_animation_data"

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [header_response, data_response]
    mock_stream.done = MagicMock(return_value=False)  # Use regular MagicMock for done()
    mock_stub.ProcessAudioStream.return_value = mock_stream

    return mock_stub


@pytest.mark.asyncio
async def test_audio2face_3d_basic_flow():
    """Test the basic flow of the Audio2Face 3D service.

    Tests:
        - TTS audio frame processing
        - Animation data stream generation
        - Frame sequence validation
        - Frame count verification

    Raises:
        AssertionError: If frame sequence or counts are incorrect.
    """
    mock_stub = get_mock_stream()

    with patch("nvidia_pipecat.services.audio2face_3d_service.A2FControllerServiceStub", return_value=mock_stub):
        # Initialize test components
        storage = FrameStorage()
        a2f = Audio2Face3DService()
        pipeline = Pipeline([a2f, storage])
        audio = bytes([6] * (16000 * 2 + 1))

        async def test_routine(task):
            # Send TTS frames
            await task.queue_frame(TTSStartedFrame())
            await task.queue_frame(TTSAudioRawFrame(audio=audio[0:15999], sample_rate=16000, num_channels=1))
            await task.queue_frame(TTSAudioRawFrame(audio=audio[16000:], sample_rate=16000, num_channels=1))
            await task.queue_frame(TTSStoppedFrame())

            # Wait for animation started frame
            started_frame = ignore(
                AnimationDataStreamStartedFrame(
                    audio_header=ANY, animation_header=ANY, animation_source_id="Audio2Face with Emotions"
                ),
                "all_ids",
                "timestamps",
            )

            await storage.wait_for_frame(started_frame)

            # Wait for animation data frame
            anim_frame = ignore(
                AnimationDataStreamRawFrame(animation_data="test_animation_data"),
                "all_ids",
                "timestamps",
            )
            await storage.wait_for_frame(anim_frame)

            # Wait for stopped frame
            stopped_frame = ignore(AnimationDataStreamStoppedFrame(), "all_ids", "timestamps")
            await storage.wait_for_frame(stopped_frame)

            # Verify the sequence of frames
            assert len(storage.frames_of_type(AnimationDataStreamStartedFrame)) == 1
            assert len(storage.frames_of_type(AnimationDataStreamRawFrame)) == 1
            assert len(storage.frames_of_type(AnimationDataStreamStoppedFrame)) == 1

        await run_interactive_test(pipeline, test_coroutine=test_routine)


@pytest.mark.asyncio
async def test_interruptions():
    """Test interruption handling in the Audio2Face 3D service.

    Verifies that the service correctly handles interruption events by:
    - Properly responding to StartInterruptionFrame
    - Sending end-of-audio signal to the A2F stream
    - Maintaining correct state during interruption
    """
    mock_stub = get_mock_stream()

    with patch("nvidia_pipecat.services.audio2face_3d_service.A2FControllerServiceStub", return_value=mock_stub):
        # Create service and storage
        storage = FrameStorage()
        a2f = Audio2Face3DService()
        pipeline = Pipeline([a2f, storage])

        async def test_routine(task):
            # Send TTS frames
            await task.queue_frame(TTSStartedFrame())
            await task.queue_frame(TTSAudioRawFrame(audio=b"test_audio", sample_rate=16000, num_channels=1))

            await asyncio.sleep(0.1)
            await task.queue_frame(StartInterruptionFrame())
            await asyncio.sleep(0.1)

            mock_stub.ProcessAudioStream.return_value.write.assert_called_with(
                AudioWithEmotionStream(end_of_audio=AudioWithEmotionStream.EndOfAudio())
            )
            print(f"{mock_stub}")
            print(f"storage.history: {storage.history}")

        await run_interactive_test(pipeline, test_coroutine=test_routine)
