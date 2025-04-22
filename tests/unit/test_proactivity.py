# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the proactivity processor module.

This module contains tests that verify the behavior of the ProactivityProcessor.
"""

import asyncio
import os
import sys

import pytest

sys.path.append(os.path.abspath("../../src"))

from pipecat.frames.frames import TTSSpeakFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask

from nvidia_pipecat.frames.action import StartedPresenceUserActionFrame
from nvidia_pipecat.processors.proactivity import ProactivityProcessor
from tests.unit.utils import FrameStorage, run_interactive_test


@pytest.mark.asyncio
async def test_proactive_bot_processor_timer_behavior():
    """Test the ProactiveBotProcessor's timer and message behavior.

    Tests the processor's ability to manage timer-based proactive messages
    and handle timer resets based on user activity.

    Args:
        None

    Returns:
        None

    The test verifies:
        - Default message is sent after timer expiration
        - Timer resets correctly on user activity
        - Timer reset prevents premature message generation
        - Frames are processed in correct order
        - Message content matches configuration
    """
    proactivity = ProactivityProcessor(default_message="I'm here if you need me!", timer_duration=0.5)
    storage = FrameStorage()
    pipeline = Pipeline([proactivity, storage])

    async def test_routine(task: PipelineTask):
        """Inner test coroutine for proactivity testing.

        Args:
            task: PipelineTask instance for frame queueing.

        The routine:
            1. Sends initial presence frame
            2. Waits for timer expiration
            3. Verifies message generation
            4. Tests timer reset behavior
            5. Confirms no premature messages
        """
        await task.queue_frame(StartedPresenceUserActionFrame(action_id="1"))
        # Wait for initial proactive message
        await asyncio.sleep(0.6)

        # Confirm at least one frame
        assert len(storage.history) >= 1, "Expected at least one frame."

        # Confirm correct text frame output
        frame = storage.history[2].frame
        assert isinstance(frame, TTSSpeakFrame)
        assert frame.text == "I'm here if you need me!"

        # Send another StartFrame to reset the timer
        await task.queue_frame(UserStoppedSpeakingFrame())
        await asyncio.sleep(0)

        # Wait half the timer (0.5s) => no new message yet
        frame_count_after_reset = len(storage.history)
        await asyncio.sleep(0.3)
        # Confirm no additional message arrived yet
        assert frame_count_after_reset == len(storage.history)

    await run_interactive_test(pipeline, test_coroutine=test_routine)
