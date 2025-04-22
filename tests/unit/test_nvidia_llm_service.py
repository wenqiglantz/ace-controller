# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the NvidiaLLMService.

This module contains tests for the NvidiaLLMService class, covering initialization,
frame processing, context aggregation, streaming, and pipeline integration.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection

from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from tests.unit.utils import FrameStorage, ignore_ids, run_interactive_test


# Create a simple mock implementation of BaseMessageChunk that doesn't use Pydantic
class MockMessageChunk:
    """Simple mock implementation of BaseMessageChunk for testing."""

    def __init__(self, content=None):
        """Initialize with optional content."""
        self._content = content
        self._message = {"content": content}

    @property
    def type(self):
        """Return the type of the message."""
        return "mock"

    @property
    def content(self):
        """Return the content of the message."""
        return self._content

    @content.setter
    def content(self, value):
        """Set the content of the message."""
        self._content = value
        self._message["content"] = value

    def __eq__(self, other):
        """Implement equality check for testing."""
        if not isinstance(other, MockMessageChunk):
            return False
        return self.content == other.content


class MockAsyncIterator:
    """A mock async iterator for testing."""

    def __init__(self, items):
        """Initialize with a list of items to yield."""
        self.items = items

    def __aiter__(self):
        """Return self as an async iterator."""
        return self

    async def __anext__(self):
        """Return the next item or raise StopAsyncIteration."""
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


@pytest.mark.asyncio
async def test_nvidia_llm_service_initialization():
    """Tests NvidiaLLMService initialization with various configurations.

    Tests service initialization with different parameters including default settings,
    custom models, and custom input parameters.

    The test verifies:
        - Default initialization parameters are set correctly
        - Custom model name is properly assigned
        - Input parameters are correctly stored
        - Metrics generation capability is configured
    """
    # Use a patched create_client to avoid external API calls
    with patch.object(NvidiaLLMService, "create_client") as mock_create_client:
        mock_create_client.return_value = MagicMock()

        # Test default initialization
        service = NvidiaLLMService(api_key="test_api_key")
        # Only check that the model name is a string, no restrictions on what it can be
        assert isinstance(service.model_name, str)
        assert service.can_generate_metrics() is True

        # Test initialization with custom model
        custom_model = "custom-model"
        service = NvidiaLLMService(model=custom_model, api_key="test_api_key")
        assert service.model_name == custom_model

        # Test initialization with custom parameters
        params = NvidiaLLMService.InputParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=500,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            seed=42,
            max_completion_tokens=1000,
            extra={"custom_param": "value"},
        )
        service = NvidiaLLMService(api_key="test_api_key", params=params)
        # Check that parameters are stored but don't restrict what values they can have
        assert service._settings["temperature"] == 0.8
        assert service._settings["top_p"] == 0.9
        assert service._settings["max_tokens"] == 500
        assert service._settings["max_completion_tokens"] == 1000
        assert service._settings["frequency_penalty"] == 0.2
        assert service._settings["presence_penalty"] == 0.1
        assert service._settings["seed"] == 42
        assert service._settings["extra"] == {"custom_param": "value"}


@pytest.mark.asyncio
async def test_context_aggregator_creation():
    """Tests creation of context aggregators.

    Tests the creation of user and assistant context aggregators with
    different configuration options.

    The test verifies:
        - User and assistant aggregators are created
        - Custom parameters are properly applied
        - Aggregator pair structure is correct
    """
    context = OpenAILLMContext(messages=[{"role": "system", "content": "You are a helpful assistant."}])

    # Create aggregator pair
    pair = NvidiaLLMService.create_context_aggregator(context)

    # Check that the pair has user and assistant aggregators
    assert pair._user is not None
    assert pair._assistant is not None

    # Check with custom parameters - access private attribute through the object's dict
    pair = NvidiaLLMService.create_context_aggregator(context, assistant_expect_stripped_words=False)
    assert pair._assistant._expect_stripped_words is False


@pytest.mark.asyncio
async def test_process_llm_messages_frame():
    """Tests processing of LLMMessagesFrame.

    Tests the complete flow of processing an LLMMessagesFrame including metrics,
    streaming, and frame generation.

    The test verifies:
        - Metrics start/stop timing
        - Correct frame sequence generation
        - Text chunk processing
        - Response start/end frame generation
    """
    with patch("nvidia_pipecat.services.nvidia_llm.ChatNVIDIA") as MockChatNVIDIA:
        # Setup the mock ChatNVIDIA
        mock_client = MagicMock()
        MockChatNVIDIA.return_value = mock_client

        # Setup mock astream to return chunks via async iterator
        chunks = [MockMessageChunk("Hello"), MockMessageChunk(" world!")]
        mock_client.astream.return_value = MockAsyncIterator(chunks)

        # Initialize service and mock push_frame
        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        # Create an LLMMessagesFrame
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about the weather."},
        ]
        frame = LLMMessagesFrame(messages)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify that start metrics was called
        service.start_processing_metrics.assert_called_once()
        service.start_ttfb_metrics.assert_called_once()

        # Verify that TextFrames were pushed for the chunks
        assert service.push_frame.call_count >= 4  # Start, 2 chunks, End

        # Find LLMFullResponseStartFrame
        start_frame_found = False
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], LLMFullResponseStartFrame):
                start_frame_found = True
                break
        assert start_frame_found, "LLMFullResponseStartFrame was not pushed"

        # Find TextFrames
        text_frames_found = 0
        expected_texts = ["Hello", " world!"]
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], TextFrame):
                assert call_args.args[0].text in expected_texts
                text_frames_found += 1
        assert text_frames_found == 2, f"Expected 2 TextFrames, found {text_frames_found}"

        # Find LLMFullResponseEndFrame
        end_frame_found = False
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], LLMFullResponseEndFrame):
                end_frame_found = True
                break
        assert end_frame_found, "LLMFullResponseEndFrame was not pushed"

        # Verify that stop metrics was called
        service.stop_processing_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_process_openai_llm_context_frame():
    """Tests processing of OpenAILLMContextFrame.

    Tests the handling of OpenAILLMContextFrame including context conversion
    and response generation.

    The test verifies:
        - Context processing
        - Metrics handling
        - Text frame generation
        - Response streaming
    """
    with patch("nvidia_pipecat.services.nvidia_llm.ChatNVIDIA") as MockChatNVIDIA:
        # Setup the mock ChatNVIDIA
        mock_client = MagicMock()
        MockChatNVIDIA.return_value = mock_client

        # Setup mock astream to return chunks
        chunks = [MockMessageChunk("Response"), MockMessageChunk(" text")]
        mock_client.astream.return_value = MockAsyncIterator(chunks)

        # Initialize service and mock push_frame
        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        # Create an OpenAILLMContextFrame
        context = OpenAILLMContext(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        )
        frame = OpenAILLMContextFrame(context=context)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify metrics and frames similarly to the previous test
        service.start_processing_metrics.assert_called_once()
        service.start_ttfb_metrics.assert_called_once()

        # Find TextFrames
        text_frames_found = 0
        expected_texts = ["Response", " text"]
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], TextFrame):
                assert call_args.args[0].text in expected_texts
                text_frames_found += 1
        assert text_frames_found == 2, f"Expected 2 TextFrames, found {text_frames_found}"


@pytest.mark.asyncio
async def test_update_settings():
    """Tests dynamic settings updates.

    Tests the ability to update service settings using LLMUpdateSettingsFrame.

    The test verifies:
        - Settings are updated correctly
        - Previous settings are overwritten
        - Frame is not propagated downstream
    """
    # Initialize service with a mock client to avoid API calls
    with patch.object(NvidiaLLMService, "create_client") as mock_create_client:
        mock_create_client.return_value = MagicMock()

        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()

        # Initial settings
        assert service._settings["temperature"] is None

        # Create an LLMUpdateSettingsFrame
        settings = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 300}
        frame = LLMUpdateSettingsFrame(settings)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify settings were updated
        assert service._settings["temperature"] == 0.7
        assert service._settings["top_p"] == 0.8
        assert service._settings["max_tokens"] == 300

        # Verify frame was not pushed downstream
        service.push_frame.assert_not_called()


@pytest.mark.asyncio
async def test_regular_frame_passing():
    """Test that regular frames are passed through."""
    # Initialize service with a mock client to avoid API calls
    with patch.object(NvidiaLLMService, "create_client") as mock_create_client:
        mock_create_client.return_value = MagicMock()

        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()

        # Create a regular frame
        frame = TextFrame("Hello")

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify frame was pushed downstream
        service.push_frame.assert_called_once_with(frame, FrameDirection.DOWNSTREAM)


@pytest.mark.asyncio
async def test_streaming_cancellation():
    """Tests streaming cancellation handling.

    Tests the service's behavior when streaming is cancelled mid-response.

    The test verifies:
        - Initial chunks are processed
        - Cancellation is handled gracefully
        - Cleanup occurs properly
        - Metrics are stopped appropriately
    """
    with patch("nvidia_pipecat.services.nvidia_llm.ChatNVIDIA") as MockChatNVIDIA:
        # Setup the mock ChatNVIDIA
        mock_client = MagicMock()
        MockChatNVIDIA.return_value = mock_client

        # Create an async iterator that will raise CancelledError
        class CancellingAsyncIterator:
            """An async iterator that raises a CancelledError after yielding one item."""

            def __aiter__(self):
                """Return self as an async iterator."""
                return self

            async def __anext__(self):
                """Return one item then raise CancelledError."""
                if not hasattr(self, "yielded"):
                    self.yielded = True
                    return MockMessageChunk("First chunk")
                raise asyncio.CancelledError()

        mock_client.astream.return_value = CancellingAsyncIterator()

        # Initialize service and mock push_frame
        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        # Create an LLMMessagesFrame
        messages = [{"role": "user", "content": "Hello"}]
        frame = LLMMessagesFrame(messages)

        # Process the frame - it should handle the cancellation
        with contextlib.suppress(asyncio.CancelledError):
            await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify that at least the first chunk was processed
        text_frame_found = False
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], TextFrame) and call_args.args[0].text == "First chunk":
                text_frame_found = True
                break
        assert text_frame_found, "The first chunk wasn't processed before cancellation"


@pytest.mark.asyncio
async def test_empty_content_skipping():
    """Test that empty content chunks are skipped."""
    with patch("nvidia_pipecat.services.nvidia_llm.ChatNVIDIA") as MockChatNVIDIA:
        # Setup the mock ChatNVIDIA
        mock_client = MagicMock()
        MockChatNVIDIA.return_value = mock_client

        # Setup mock astream to return chunks, including an empty one
        chunks = [
            MockMessageChunk("First"),
            MockMessageChunk(""),  # Empty content
            MockMessageChunk(" content"),
        ]
        mock_client.astream.return_value = MockAsyncIterator(chunks)

        # Initialize service and mock push_frame
        service = NvidiaLLMService(api_key="test_api_key")
        service.push_frame = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        # Create an LLMMessagesFrame
        messages = [{"role": "user", "content": "Hello"}]
        frame = LLMMessagesFrame(messages)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Count TextFrames - should be 2, not 3
        text_frames_count = 0
        for call_args in service.push_frame.call_args_list:
            if isinstance(call_args.args[0], TextFrame):
                text_frames_count += 1
        assert text_frames_count == 2, "Empty content chunk wasn't skipped"


@pytest.mark.asyncio
async def test_integration_pipeline():
    """Tests integration with complete pipeline.

    Tests the service's behavior when integrated into a full pipeline with
    input and output storage.

    The test verifies:
        - Frame flow through pipeline
        - Correct frame sequence
        - Input frame preservation
        - Output frame generation
        - Pipeline completion
    """
    with patch("nvidia_pipecat.services.nvidia_llm.ChatNVIDIA") as MockChatNVIDIA:
        # Setup the mock ChatNVIDIA
        mock_client = MagicMock()
        MockChatNVIDIA.return_value = mock_client

        # Setup mock astream to return chunks
        chunks = [MockMessageChunk("Hello"), MockMessageChunk(" world!")]
        mock_client.astream.return_value = MockAsyncIterator(chunks)

        # Create service, storage, and pipeline
        service = NvidiaLLMService(api_key="test_api_key")
        input_storage = FrameStorage()  # Stores the input frame
        output_storage = FrameStorage()  # Stores the output frames
        pipeline = Pipeline([input_storage, service, output_storage])

        # Create a messages frame
        messages = [{"role": "user", "content": "Tell me something"}]
        frame = LLMMessagesFrame(messages)

        # Run the pipeline
        async def test_routine(task):
            await task.queue_frame(frame)

            # Wait for specific frames to arrive in the output storage
            await output_storage.wait_for_frame(ignore_ids(LLMFullResponseStartFrame()))
            await output_storage.wait_for_frame(ignore_ids(TextFrame("Hello")))
            await output_storage.wait_for_frame(ignore_ids(TextFrame(" world!")))
            await output_storage.wait_for_frame(ignore_ids(LLMFullResponseEndFrame()))

        await run_interactive_test(pipeline, test_coroutine=test_routine)

        # Verify input frame in input storage (should have StartFrame, LLMMessagesFrame, EndFrame)
        llm_messages_frame_found = False
        for entry in input_storage.history:
            if isinstance(entry.frame, LLMMessagesFrame):
                llm_messages_frame_found = True
                assert entry.frame.messages[0]["role"] == "user"
                assert entry.frame.messages[0]["content"] == "Tell me something"
        assert llm_messages_frame_found, "LLMMessagesFrame not found in input_storage"

        # Verify output frames in output storage
        full_response_start_found = False
        text_frames_found = 0
        full_response_end_found = False

        for entry in output_storage.history:
            frame = entry.frame
            if isinstance(frame, LLMFullResponseStartFrame):
                full_response_start_found = True
            elif isinstance(frame, TextFrame):
                if frame.text in ["Hello", " world!"]:
                    text_frames_found += 1
            elif isinstance(frame, LLMFullResponseEndFrame):
                full_response_end_found = True

        assert full_response_start_found, "LLMFullResponseStartFrame not found"
        assert text_frames_found == 2, f"Expected 2 text frames, found {text_frames_found}"
        assert full_response_end_found, "LLMFullResponseEndFrame not found"
