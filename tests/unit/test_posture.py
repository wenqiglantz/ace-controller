# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the PostureProviderProcessor class."""

import pytest
from pipecat.frames.frames import BotStoppedSpeakingFrame, StartInterruptionFrame, TTSStartedFrame

from nvidia_pipecat.frames.action import StartPostureBotActionFrame
from nvidia_pipecat.processors.posture_provider import PostureProviderProcessor
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio()
async def test_posture_provider_processor_tts():
    """Test TTSStartedFrame processing in PostureProviderProcessor.

    Tests that the processor generates appropriate "Talking" posture when
    text-to-speech begins.

    Args:
        None

    Returns:
        None

    The test verifies:
        - TTSStartedFrame is processed correctly
        - "Talking" posture is generated
        - Frames are emitted in correct order
    """
    frames_to_send = [TTSStartedFrame()]
    expected_down_frames = [
        ignore_ids(TTSStartedFrame()),
        ignore_ids(StartPostureBotActionFrame(posture="Talking")),
    ]

    await run_test(
        PostureProviderProcessor(),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_posture_provider_processor_bot_finished():
    """Test BotStoppedSpeakingFrame processing in PostureProviderProcessor.

    Tests that the processor generates appropriate "Attentive" posture when
    bot stops speaking.

    Args:
        None

    Returns:
        None

    The test verifies:
        - BotStoppedSpeakingFrame is processed correctly
        - "Attentive" posture is generated
        - Frames are emitted in correct order
    """
    frames_to_send = [BotStoppedSpeakingFrame()]
    expected_down_frames = [
        ignore_ids(BotStoppedSpeakingFrame()),
        ignore_ids(StartPostureBotActionFrame(posture="Attentive")),
    ]

    await run_test(
        PostureProviderProcessor(),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )


@pytest.mark.asyncio()
async def test_posture_provider_processor_interrupt():
    """Tests posture generation for interruption events.

    Tests that the processor generates appropriate "Listening" posture when
    an interruption occurs.

    Args:
        None

    Returns:
        None

    The test verifies:
        - StartInterruptionFrame is processed correctly
        - "Listening" posture is generated
        - Frames are emitted in correct order
    """
    frames_to_send = [StartInterruptionFrame()]
    expected_down_frames = [
        ignore_ids(StartInterruptionFrame()),
        ignore_ids(StartPostureBotActionFrame(posture="Listening")),
    ]

    await run_test(
        PostureProviderProcessor(),
        frames_to_send=frames_to_send,
        expected_down_frames=expected_down_frames,
    )
