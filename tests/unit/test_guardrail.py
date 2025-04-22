# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the user presence frame processor."""

import pytest
from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.utils.time import time_now_iso8601

from nvidia_pipecat.processors.guardrail import GuardrailProcessor
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio
async def test_blocked_word():
    """Tests blocking functionality for explicitly blocked words.

    Tests that the processor replaces transcription frames containing blocked words
    with a TTSSpeakFrame containing a rejection message.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Input containing "football" is blocked
        - Response is replaced with rejection message
    """
    guardrail_bot = GuardrailProcessor(blocked_words=["football"])
    frames_to_send = [TranscriptionFrame(text="I love football", user_id="", timestamp=time_now_iso8601())]
    expected_down_frames = [ignore_ids(TTSSpeakFrame("I am not allowed to answer this question"))]

    await run_test(guardrail_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_non_blocked_word():
    """Tests passthrough of allowed words.

    Tests that the processor allows transcription frames that don't contain
    any blocked words to pass through unchanged.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Input without blocked words passes through
        - Frame content remains unchanged
    """
    guardrail_bot = GuardrailProcessor(blocked_words=["football"])
    timestamp = time_now_iso8601()
    frames_to_send = [TranscriptionFrame(text="Tell me about Pasta", user_id="", timestamp=timestamp)]
    expected_down_frames = [ignore_ids(TranscriptionFrame(text="Tell me about Pasta", user_id="", timestamp=timestamp))]

    await run_test(guardrail_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_substring_blocked_word():
    """Tests substring matching behavior for blocked words.

    Tests that the processor only blocks exact word matches and allows
    words that contain blocked words as substrings.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Words containing blocked words as substrings are allowed
        - Frame content remains unchanged
    """
    guardrail_bot = GuardrailProcessor(blocked_words=["foot"])
    timestamp = time_now_iso8601()
    frames_to_send = [TranscriptionFrame(text="I love football", user_id="", timestamp=timestamp)]
    expected_down_frames = [ignore_ids(TranscriptionFrame(text="I love football", user_id="", timestamp=timestamp))]

    await run_test(guardrail_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_no_blocked_word():
    """Tests default behavior with no blocked words configured.

    Tests that the processor allows all transcription frames to pass through
    when no blocked words are specified.

    Args:
        None

    Returns:
        None

    The test verifies:
        - All input passes through when no words are blocked
        - Frame content remains unchanged
    """
    guardrail_bot = GuardrailProcessor()
    timestamp = time_now_iso8601()
    frames_to_send = [TranscriptionFrame(text="What is your name", user_id="", timestamp=timestamp)]
    expected_down_frames = [ignore_ids(TranscriptionFrame(text="What is your name", user_id="", timestamp=timestamp))]

    await run_test(guardrail_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)
