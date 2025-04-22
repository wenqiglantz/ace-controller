# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Nvidia Aggregators."""

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as run_pipecat_test
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.frames.riva import RivaInterimTranscriptionFrame
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaUserContextAggregator,
    create_nvidia_context_aggregator,
)


@pytest.mark.asyncio()
async def test_normal_flow():
    """Test the normal flow of user and assistant interactions with interim transcriptions enabled.

    Tests the sequence of events from user speech start through assistant response,
    verifying proper handling of interim and final transcriptions.

    The test verifies:
        - User speech start frame handling
        - Interim transcription processing
        - Final transcription handling
        - User speech stop frame handling
        - Assistant response processing
        - Context updates at each stage
    """
    messages = []
    context = OpenAILLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})
    # Test Case 1: Normal flow with UserStartedSpeakingFrame first
    frames_to_send = [
        UserStartedSpeakingFrame(),
        LLMMessagesFrame(messages),
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        TranscriptionFrame("Hello User Aggregator!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]
    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        LLMMessagesFrame,
        OpenAILLMContextFrame,  # From first interim
        OpenAILLMContextFrame,  # From first final
        UserStoppedSpeakingFrame,
        OpenAILLMContextFrame,
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Hello User Aggregator!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_user_speaking_frame_delay_cases():
    """Test handling of transcription frames that arrive before UserStartedSpeakingFrame.

    Tests edge cases around transcription frame timing relative to the
    UserStartedSpeakingFrame.

    The test verifies:
        - Interim frames before UserStartedSpeakingFrame are ignored
        - Low stability interim frames are ignored
        - Only processes transcriptions after UserStartedSpeakingFrame
        - Context is updated correctly for valid frames
    """
    messages = []
    context = OpenAILLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 2: RivaInterimTranscriptionFrames before UserStartedSpeakingFrame
    frames_to_send = [
        RivaInterimTranscriptionFrame(
            "Testing", "", time_now_iso8601(), None, stability=0.5
        ),  # Should be ignored (low stability)
        RivaInterimTranscriptionFrame(
            "Testing delayed", "", time_now_iso8601(), None, stability=1.0
        ),  # Should be ignored (no UserStartedSpeakingFrame yet)
        SleepFrame(0.5),
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame(
            "Testing after start", "", time_now_iso8601(), None, stability=1.0
        ),  # Should be processed
        TranscriptionFrame("Testing after start complete", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        OpenAILLMContextFrame,  # from first interim after UserStartedSpeakingFrame
        OpenAILLMContextFrame,  # From first final after UserStartedSpeakingFrame
        UserStoppedSpeakingFrame,
        OpenAILLMContextFrame,
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Testing after start complete"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_multiple_interims_with_final_transcription():
    """Test handling of multiple interim transcription frames followed by a final transcription.

    Tests the processing of a sequence of interim transcriptions followed by
    a final transcription.

    The test verifies:
        - Multiple interim transcriptions are processed correctly
        - Final transcription properly overwrites previous interims
        - Context updates occur for each valid frame
        - Message history maintains correct order
    """
    messages = []
    context = OpenAILLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 3: Multiple interim frames with final transcription
    frames_to_send = [
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello Again", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello Again User", "", time_now_iso8601(), None, stability=1.0),
        TranscriptionFrame("Hello Again User Aggregator!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )
    expected_down_frames = [
        UserStartedSpeakingFrame,
        OpenAILLMContextFrame,  # from first interim
        OpenAILLMContextFrame,  # From second interim
        OpenAILLMContextFrame,  # From third interim
        OpenAILLMContextFrame,  # From final transcription
        UserStoppedSpeakingFrame,
        OpenAILLMContextFrame,
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Hello Again User Aggregator!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_transcription_after_user_stopped_speaking():
    """Tests handling of late transcription frames.

    Tests behavior when transcription frames arrive after UserStoppedSpeakingFrame.

    The test verifies:
        - Late transcriptions are still processed
        - Context is updated with final transcription
        - Assistant responses are handled correctly
        - Message history maintains proper sequence
    """
    messages = []
    context = OpenAILLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)

    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])
    messages.append({"role": "system", "content": "This is system prompt"})

    # Test Case 4: TranscriptionFrame after UserStoppedSpeakingFrame
    frames_to_send = [
        UserStartedSpeakingFrame(),
        RivaInterimTranscriptionFrame("Late", "", time_now_iso8601(), None, stability=1.0),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
        TranscriptionFrame("Late transcription!", 1, 2),
        SleepFrame(0.1),
    ]

    # Assistant response
    frames_to_send.extend(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello Assistant Aggregator!"),
            LLMFullResponseEndFrame(),
        ]
    )

    expected_down_frames = [
        UserStartedSpeakingFrame,
        OpenAILLMContextFrame,  # From first interim
        UserStoppedSpeakingFrame,
        OpenAILLMContextFrame,  # From final after UserStoppedSpeakingFrame
        OpenAILLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "user", "content": "Late transcription!"},
        {"role": "assistant", "content": "Hello Assistant Aggregator!"},
    ]


@pytest.mark.asyncio()
async def test_no_interim_frames():
    """Tests behavior when interim frames are disabled.

    Tests the aggregator's handling of transcriptions when send_interims=False.

    The test verifies:
        - Interim frames are ignored
        - Only final transcription is processed
        - System prompts are preserved
        - Context updates occur only for final transcription
        - Assistant responses are processed correctly
    """
    messages = [{"role": "system", "content": "This is system prompt"}]
    context = OpenAILLMContext(messages)
    context_aggregator = create_nvidia_context_aggregator(context, send_interims=False)
    pipeline = Pipeline([context_aggregator.user(), context_aggregator.assistant()])

    frames_to_send = [
        UserStartedSpeakingFrame(),
        LLMMessagesFrame(messages),
        # These interim frames should be ignored due to send_interims=False
        RivaInterimTranscriptionFrame("Hello", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello there", "", time_now_iso8601(), None, stability=1.0),
        RivaInterimTranscriptionFrame("Hello there user", "", time_now_iso8601(), None, stability=1.0),
        # Only the final transcription should be processed
        TranscriptionFrame("Hello there user final!", 1, 2),
        SleepFrame(0.1),
        UserStoppedSpeakingFrame(),
        SleepFrame(0.1),
        # Assistant response
        LLMFullResponseStartFrame(),
        TextFrame("Hello from assistant!"),
        LLMFullResponseEndFrame(),
    ]

    expected_down_frames = [
        UserStartedSpeakingFrame,
        LLMMessagesFrame,
        OpenAILLMContextFrame,  # Only from final transcription
        UserStoppedSpeakingFrame,
        OpenAILLMContextFrame,  # From assistant response
    ]

    await run_pipecat_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )

    # Verify final context state
    assert context_aggregator.user().context.get_messages() == [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "Hello there user final!"},
        {"role": "assistant", "content": "Hello from assistant!"},
    ]


@pytest.mark.asyncio()
async def test_get_truncated_context():
    """Tests context truncation functionality.

    Tests the get_truncated_context() method of NvidiaUserContextAggregator
    with a specified chat history limit.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Context is truncated to specified limit
        - System prompt is preserved
        - Most recent messages are retained
        - Message order is maintained
    """
    messages = [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "Hi, there!"},
        {"role": "assistant", "content": "Hello, how may I assist you?"},
        {"role": "user", "content": "How to be more productive?"},
        {"role": "assistant", "content": "Priotize the tasks, make a list..."},
        {"role": "user", "content": "What is metaverse?"},
        {
            "role": "assistant",
            "content": "The metaverse is envisioned as a digital ecosystem built on virtual 3D technology",
        },
        {
            "role": "assistant",
            "content": "It leverages 3D technology and digital"
            "representation for creating virtual environments and user experiences",
        },
        {"role": "user", "content": "thanks, Bye!"},
    ]
    context = OpenAILLMContext(messages)
    user = NvidiaUserContextAggregator(context=context, chat_history_limit=2)
    truncated_context = await user.get_truncated_context()
    assert truncated_context.get_messages() == [
        {"role": "system", "content": "This is system prompt"},
        {"role": "user", "content": "What is metaverse?"},
        {
            "role": "assistant",
            "content": "The metaverse is envisioned as a digital ecosystem built on virtual 3D technology",
        },
        {
            "role": "assistant",
            "content": "It leverages 3D technology and digital"
            "representation for creating virtual environments and user experiences",
        },
        {"role": "user", "content": "thanks, Bye!"},
    ]
