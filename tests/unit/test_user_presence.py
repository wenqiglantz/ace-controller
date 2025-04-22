# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the user presence frame processor."""

import pytest
from pipecat.frames.frames import (
    FilterControlFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame

from nvidia_pipecat.frames.action import FinishedPresenceUserActionFrame, StartedPresenceUserActionFrame
from nvidia_pipecat.processors.user_presence import UserPresenceProcesssor
from tests.unit.utils import ignore, run_test


@pytest.mark.asyncio
async def test_user_presence_start():
    """Tests user presence start handling with welcome message.

    Tests that the processor correctly handles user presence start events
    and generates appropriate welcome messages.

    Args:
        None

    Returns:
        None

    The test verifies:
        - StartedPresenceUserActionFrame processing
        - Welcome message generation
        - UserStartedSpeakingFrame handling
        - Frame sequence ordering
        - Message content accuracy
    """
    user_presence_bot = UserPresenceProcesssor(welcome_msg="Hey there!")
    frames_to_send = [StartedPresenceUserActionFrame(action_id=123), SleepFrame(0.01), UserStartedSpeakingFrame()]
    expected_down_frames = [
        ignore(StartedPresenceUserActionFrame(action_id=123), "ids", "timestamps"),
        ignore(TTSSpeakFrame("Hey there!"), "ids"),
        ignore(UserStartedSpeakingFrame(), "ids", "timestamps"),
    ]

    await run_test(user_presence_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_user_presence_finished():
    """Tests user presence finish handling with farewell message.

    Tests that the processor correctly handles user presence finish events
    and generates appropriate farewell messages.

    Args:
        None

    Returns:
        None

    The test verifies:
        - StartedPresenceUserActionFrame processing
        - FinishedPresenceUserActionFrame handling
        - Farewell message generation
        - StartInterruptionFrame sequencing
        - Frame ordering
        - Message content accuracy
    """
    user_presence_bot = UserPresenceProcesssor(farewell_msg="Bye bye!")
    frames_to_send = [
        StartedPresenceUserActionFrame(action_id=123),
        SleepFrame(0.5),
        FinishedPresenceUserActionFrame(action_id=123),
        SleepFrame(0.5),
    ]

    expected_down_frames = [
        ignore(StartedPresenceUserActionFrame(action_id=123), "ids", "timestamps"),
        ignore(TTSSpeakFrame("Hello"), "ids"),
        # WAR: The StartInterruptionFrame is sent in response to the FinishedPresenceUserActionFrame.
        # However, the test framework consistently logs it in the sequence prior to the FinishedPresenceUserFrame
        ignore(StartInterruptionFrame(), "ids", "timestamps"),
        ignore(FinishedPresenceUserActionFrame(action_id=123), "ids", "timestamps"),
        ignore(TTSSpeakFrame("Bye bye!"), "ids"),
    ]

    await run_test(user_presence_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_user_presence():
    """Tests behavior without presence frames.

    Tests that no welcome/farewell messages are sent when no presence
    frames are received.

    Args:
        None

    Returns:
        None

    The test verifies:
        - No messages without presence frames
        - UserStartedSpeakingFrame handling
        - Frame filtering behavior
    """
    user_presence_bot = UserPresenceProcesssor(welcome_msg="Hello", farewell_msg="Bye bye!")
    frames_to_send = [UserStartedSpeakingFrame()]
    expected_down_frames = []

    await run_test(user_presence_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)


@pytest.mark.asyncio
async def test_user_presence_system_frames():
    """Tests system and control frame handling.

    Tests that system and control frames are processed regardless of
    user presence state.

    Args:
        None

    Returns:
        None

    The test verifies:
        - TTSStartedFrame processing
        - FilterControlFrame handling
        - LLMUpdateSettingsFrame processing
        - TextFrame filtering
        - Frame passthrough behavior
        - Frame sequence preservation
    """
    user_presence_bot = UserPresenceProcesssor(welcome_msg="Hello", farewell_msg="Bye bye!")

    frames_to_send = [
        TTSStartedFrame(),
        FilterControlFrame(),
        LLMUpdateSettingsFrame(settings="ABC"),
        TextFrame("How are you?"),
    ]

    expected_down_frames = [
        ignore(TTSStartedFrame(), "ids", "timestamps"),
        ignore(FilterControlFrame(), "ids", "timestamps"),
        ignore(LLMUpdateSettingsFrame(settings="ABC"), "ids", "timestamps"),
    ]

    await run_test(user_presence_bot, frames_to_send=frames_to_send, expected_down_frames=expected_down_frames)
