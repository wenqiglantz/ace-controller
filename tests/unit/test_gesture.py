# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the FacialGestureProviderProcessor."""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    StartInterruptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame

from nvidia_pipecat.frames.action import StartFacialGestureBotActionFrame
from nvidia_pipecat.processors.gesture_provider import FacialGestureProviderProcessor
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio()
async def test_gesture_provider_processor_interrupt():
    """Test facial gesture generation for bot speech start and interruption.

    Tests that the processor generates appropriate facial gestures when receiving
    BotStartedSpeakingFrame and StartInterruptionFrame events.

    The test verifies:
        - Correct handling of BotStartedSpeakingFrame
        - Correct handling of StartInterruptionFrame
        - Generation of "Pensive" facial gesture
    """
    frames_to_send = [
        BotStartedSpeakingFrame(),
        SleepFrame(0.1),
        StartInterruptionFrame(),
    ]
    expected_down_frames = [
        ignore_ids(BotStartedSpeakingFrame()),
        ignore_ids(StartInterruptionFrame()),
        ignore_ids(StartFacialGestureBotActionFrame(facial_gesture="Pensive")),
    ]

    await run_test(
        FacialGestureProviderProcessor(probability=1.0),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_gesture_provider_processor_bot_finished():
    """Test facial gesture processing for bot speech completion.

    Tests that the processor handles BotStoppedSpeakingFrame by passing it through
    without generating additional gestures.

    The test verifies:
        - Correct passthrough of BotStoppedSpeakingFrame
        - No additional gesture generation
    """
    frames_to_send = [BotStoppedSpeakingFrame()]
    expected_down_frames = [
        ignore_ids(BotStoppedSpeakingFrame()),
    ]

    await run_test(
        FacialGestureProviderProcessor(probability=1.0),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_gesture_provider_processor_tts():
    """Test facial gesture processing for interruption events.

    Tests that the processor generates a "Taunt" facial gesture when receiving
    UserStoppedSpeakingFrame events.

    The test verifies:
        - Correct handling of UserStoppedSpeakingFrame
        - Generation of "Taunt" facial gesture
    """
    frames_to_send = [UserStoppedSpeakingFrame()]
    expected_down_frames = [
        ignore_ids(UserStoppedSpeakingFrame()),
        ignore_ids(StartFacialGestureBotActionFrame(facial_gesture="Taunt")),
    ]

    await run_test(
        FacialGestureProviderProcessor(probability=1.0),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )
