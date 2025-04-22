# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the AcknowledgmentProcessor class.

Tests the processor's ability to:
    - Generate filler responses during user pauses
    - Handle presence detection
    - Process user speech events
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
from nvidia_pipecat.processors.acknowledgment import AcknowledgmentProcessor
from tests.unit.utils import FrameStorage, run_interactive_test


@pytest.mark.asyncio
async def test_proactive_bot_processor_timer_behavior():
    """Test the ProactiveBotProcessor.

    Tests:
        - Filler word generation after user stops speaking
        - Proper handling of user presence events
        - Verification of generated responses

    Raises:
        AssertionError: If processor behavior doesn't match expected outcomes.
    """
    filler_words = ["Great question.", "Let me check.", "Hmmm"] + [""]
    filler = AcknowledgmentProcessor(filler_words=filler_words)
    storage = FrameStorage()
    pipeline = Pipeline([filler, storage])

    async def test_routine(task: PipelineTask):
        # Signal user presence
        await task.queue_frame(StartedPresenceUserActionFrame(action_id="1"))
        # Let the pipeline process presence
        await asyncio.sleep(0)

        # Signal end of user speech
        await task.queue_frame(UserStoppedSpeakingFrame())
        # Let the pipeline process the new frame and generate a TTS filler
        await asyncio.sleep(0.1)

        # Check what frames have arrived in storage
        frames = [entry.frame for entry in storage.history]

        # We expect at least one TTSSpeakFrame in there
        tts_frames = [f for f in frames if isinstance(f, TTSSpeakFrame)]
        assert len(tts_frames) > 0, "Expected a TTSSpeakFrame but found none."

        # Verify filler word content
        filler_frame = tts_frames[-1]
        # Make sure its text is one of the possible filler words
        assert filler_frame.text in filler_words, (
            f"Filler text '{filler_frame.text}' not in the list of expected filler words {filler_words}"
        )

    await run_interactive_test(pipeline, test_coroutine=test_routine)
