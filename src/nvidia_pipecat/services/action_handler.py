# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Action handler implementation for managing action state machines.

This module provides the core functionality for handling action state management through
an asynchronous state machine implementation. It defines the base ActionHandler class that
concrete action implementations can inherit from to manage their state lifecycle.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from pipecat.frames.frames import SystemFrame
from pipecat.processors.frame_processor import FrameProcessor
from transitions.extensions import MachineFactory

from nvidia_pipecat.frames.action import ActionFrame

async_machine_cls = MachineFactory.get_predefined(asyncio=True)


@dataclass
class InternalStateMachineTriggerFrame(SystemFrame):
    """Internal event frame for state machine transitions.

    Attributes:
        trigger (str): Name of the state machine trigger.
        action_name (str): Name of the action being triggered.
        action_id (str): Unique identifier for the action instance.
        data (dict[str, Any]): Additional trigger data.
    """

    trigger: str
    action_name: str = field(kw_only=True)
    action_id: str = field(kw_only=True)
    data: dict[str, Any] = field(kw_only=True, default_factory=dict)


@dataclass(kw_only=True)
class InternalStateMachineAbortFrame(InternalStateMachineTriggerFrame):
    """Internal event frame for aborting actions.

    Attributes:
        trigger (str): Always "abort" for this frame type.
        reason (str): Reason for the abort action.
    """

    trigger: str = "abort"
    reason: str = field(kw_only=True)


class ActionHandler:
    """Base class for action state machine implementations.

    Manages state transitions for actions through an async state machine. Concrete
    action implementations inherit from this class to define specific behavior.

    Attributes:
        parent_processor (FrameProcessor): The parent frame processor.
        machine (AsyncMachine): The state machine instance.
        action_state (dict[str, Any]): Current action state data.
        action_is_success (bool): Whether the action completed successfully.
        was_stopped (bool): Whether the action was explicitly stopped.
        action_failure_reason (str): Reason for action failure if unsuccessful.

    States:
        init: Initial state
        scheduled: Action is scheduled
        starting: Action is starting
        running: Action is running
        paused: Action is paused
        resuming: Action is resuming
        stopping: Action is stopping
        finished: Action is complete
    """

    states = ["init", "scheduled", "starting", "running", "paused", "resuming", "stopping", "finished"]

    def __init__(self, parent_processor: FrameProcessor) -> None:
        """Initialize the action handler.

        Args:
            parent_processor: The parent frame processor.
        """
        self.parent_processor: FrameProcessor = parent_processor
        transitions = [
            {
                "trigger": "start",
                "source": "init",
                "dest": "scheduled",
                "before": "update_action_state",
            },
            {
                "trigger": "modality_available",
                "source": "scheduled",
                "dest": "starting",
                "before": "update_action_state",
            },
            {
                "trigger": "abort",
                "source": "scheduled",
                "dest": "stopping",
                "before": ["update_action_state", "before_aborted"],
            },
            {
                "trigger": "stop",
                "source": "scheduled",
                "dest": "stopping",
                "before": ["update_action_state", "received_stop_frame"],
            },
            {
                "trigger": "pause",
                "source": "scheduled",
                "dest": "paused",
                "before": "update_action_state",
            },
            {
                "trigger": "started",
                "source": "starting",
                "dest": "running",
                "before": "update_action_state",
            },
            {
                "trigger": "stop",
                "source": "starting",
                "dest": "stopping",
                "before": [
                    "update_action_state",
                    "received_stop_frame",
                    "stopped_during_starting",
                ],
            },
            {
                "trigger": "abort",
                "source": "starting",
                "dest": "stopping",
                "before": ["update_action_state", "before_aborted"],
            },
            {
                "trigger": "abort",
                "source": "init",
                "dest": "stopping",
                "before": ["update_action_state", "before_aborted"],
            },
            {
                "trigger": "abort",
                "source": "running",
                "dest": "stopping",
                "before": ["update_action_state", "before_aborted"],
            },
            {
                "trigger": "abort",
                "source": "stopping",
                "dest": None,
                "before": [],
            },
            {
                "trigger": "change",
                "source": "running",
                "dest": "running",
                "before": ["update_action_state", "on_change"],
            },
            {
                "trigger": "pause",
                "source": "running",
                "dest": "paused",
                "before": "update_action_state",
            },
            {
                "trigger": "resume",
                "source": "paused",
                "dest": "resuming",
                "before": "update_action_state",
            },
            {
                "trigger": "stop",
                "source": "paused",
                "dest": "stopping",
                "before": [
                    "update_action_state",
                    "received_stop_frame",
                    "stopped_during_paused",
                ],
            },
            {
                "trigger": "resumed",
                "source": "resuming",
                "dest": "running",
                "before": "update_action_state",
            },
            {
                "trigger": "stop",
                "source": "running",
                "dest": "stopping",
                "before": [
                    "update_action_state",
                    "received_stop_frame",
                    "stopped_during_running",
                ],
            },
            {
                "trigger": "finished",
                "source": "stopping",
                "dest": "finished",
                "before": "update_action_state",
            },
            {"trigger": "started", "source": "stopping", "dest": None},
        ]
        self.machine = async_machine_cls(
            model=self,
            states=ActionHandler.states,
            initial="init",
            transitions=transitions,
        )
        self.action_state: dict[str, Any] = {}
        self.action_is_success: bool = False
        self.was_stopped = False
        self.action_failure_reason: str = ""

    @property
    def action_id(self) -> str:
        """Get the current action id."""
        return self.action_state.get("action_id", "")

    async def stopped_during_starting(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Handles stop event during starting state.

        Args:
            frame: The trigger or action frame that caused the stop.
        """
        if isinstance(frame, ActionFrame):
            self.action_is_success = False
            self.action_failure_reason = "Stopped during starting the action."

    async def stopped_during_running(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when the action is stopped during the running state."""
        if isinstance(frame, ActionFrame):
            self.action_is_success = True
            self.action_failure_reason = ""

    async def stopped_during_paused(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when the action is stopped during the paused state."""
        if isinstance(frame, ActionFrame):
            self.action_is_success = True
            self.action_failure_reason = ""

    async def received_stop_frame(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when a stop frame is received."""
        self.was_stopped = True

    async def before_aborted(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called before the action is aborted."""
        assert isinstance(frame, InternalStateMachineAbortFrame)
        self.action_is_success = False
        self.action_failure_reason = frame.reason

    async def update_action_state(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Updates action state data during transitions.

        Args:
            frame: The frame containing updated state data.
        """
        if isinstance(frame, ActionFrame):
            assert "action_id" not in self.action_state or frame.action_id == self.action_state["action_id"]
            self.action_state.update(asdict(frame))

    async def on_enter_running(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when the action enters the running state."""
        pass

    async def on_enter_finished(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when the action enters the finished state."""
        pass

    async def on_enter_stopping(self, frame: InternalStateMachineTriggerFrame | ActionFrame) -> None:
        """Called when the action enters the stopping state."""
        pass

    async def on_enter_starting(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> None:
        """Called when the action enters the starting state."""
        pass

    async def clear_modality(self) -> None:
        """Clear the modality for the action."""
        pass
