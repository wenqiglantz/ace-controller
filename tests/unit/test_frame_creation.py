# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for action frame creation and manipulation.

This module tests the creation, validation, and comparison of various action frames used for bot and user actions.
"""

# ruff: noqa: F405

from typing import Any

import pytest
from pipecat.frames.frames import TextFrame

from nvidia_pipecat.frames.action import *  # noqa: F403
from nvidia_pipecat.frames.action import (
    FinishedFacialGestureBotActionFrame,
    StartedFacialGestureBotActionFrame,
    StartFacialGestureBotActionFrame,
    StopFacialGestureBotActionFrame,
)
from tests.unit.utils import ignore_ids


def test_action_frame_basic_usage():
    """Tests basic action frame functionality.

    Tests:
        - Frame creation with parameters
        - Action ID propagation
        - Frame name generation
        - Frame attribute access

    Raises:
        AssertionError: If frame attributes don't match expected values.
    """
    start_frame = StartFacialGestureBotActionFrame(facial_gesture="wink")
    action_id = start_frame.action_id

    started_frame = StartedFacialGestureBotActionFrame(action_id=action_id)
    stop_frame = StopFacialGestureBotActionFrame(action_id=action_id)
    finished_frame = FinishedFacialGestureBotActionFrame(action_id=action_id)

    assert started_frame.action_id == action_id
    assert stop_frame.action_id == action_id
    assert stop_frame.name == "StopFacialGestureBotActionFrame#0"
    assert finished_frame.action_id == action_id
    assert finished_frame.name == "FinishedFacialGestureBotActionFrame#0"


def test_required_parameters():
    """Tests parameter validation in frame creation.

    Tests:
        - Required parameter enforcement
        - Type checking
        - Error handling

    Raises:
        TypeError: When required parameters are missing.
    """
    with pytest.raises(TypeError):
        StartedFacialGestureBotActionFrame()  # type: ignore


def test_action_frame_existence():
    """Tests frame class contract compliance.

    Tests:
        - Frame class initialization
        - Parameter handling
        - Action ID validation
        - Frame type verification

    Raises:
        TypeError: When frame initialization fails.
        AssertionError: If frame attributes are incorrect.
    """
    actions: dict[str, dict[str, dict[str, Any]]] = {
        "FacialGestureBotActionFrame": {
            "Start": {"facial_gesture": "wink"},
            "Started": {},
            "Stop": {},
            "Finished": {},
        },
        "GestureBotActionFrame": {
            "Start": {"gesture": "wave"},
            "Started": {},
            "Stop": {},
            "Finished": {},
        },
        "PostureBotActionFrame": {
            "Start": {"posture": "listening"},
            "Started": {},
            "Stop": {},
            "Finished": {},
        },
        "PositionBotActionFrame": {
            "Start": {"position": "left"},
            "Started": {},
            "Stop": {},
            "Updated": {"position_reached": "left"},
            "Finished": {},
        },
        "AttentionUserActionFrame": {
            "Started": {"attention_level": "attentive"},
            "Updated": {"attention_level": "inattentive"},
            "Finished": {},
        },
        "PresenceUserActionFrame": {
            "Started": {},
            "Finished": {},
        },
    }

    for action_name, frame_type in actions.items():
        for f, args in frame_type.items():
            if f == "Start":
                test = globals()[f"{f}{action_name}"](**args)
                assert test.name
            else:
                with pytest.raises(TypeError):
                    test = globals()[f"{f}{action_name}"]()

                args["action_id"] = "1234"
                test = globals()[f"{f}{action_name}"](**args)
                assert test.action_id == "1234"


def test_frame_comparison_ignoring_ids():
    """Tests frame comparison with ID ignoring.

    Tests:
        - Frame equality comparison
        - ID-independent comparison
        - Content-based comparison

    Raises:
        AssertionError: If frame comparison results are incorrect.
    """
    a = TextFrame(text="test")
    b = TextFrame(text="test")
    c = TextFrame(text="something")

    assert a != b
    assert a == ignore_ids(b)
    assert a != ignore_ids(c)
