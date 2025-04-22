# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Base classes for action services and modality management.

This module provides the foundation for implementing action services.
It includes:
- Base classes for action services and modality managers
- Support for different modality policies (Override, Parallel, Replace)
- Action state machine management
"""

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.action import (
    ActionFrame,
    ChangeActionFrame,
    FinishedActionFrame,
    StartActionFrame,
    StartedActionFrame,
    StopActionFrame,
    UpdatedActionFrame,
)
from nvidia_pipecat.services.action_handler import (
    ActionHandler,
    InternalStateMachineAbortFrame,
    InternalStateMachineTriggerFrame,
)


def short_id(id) -> str:
    """Returns shortened form of an ID.

    Args:
        id: ID to shorten.

    Returns:
        Shortened ID (first 5 chars + '..') if longer than 5 chars,
        otherwise original ID.
    """
    return f"{id[:5]}.." if len(id) > 5 else id


class ModalityManager:
    """Base class for UMIM action modality managers.

    Manages the interaction between different modalities (e.g., speech, gestures)
    and handles the state transitions of actions within each modality.

    Attributes:
        frame_types_to_process: Frame types this manager processes.
        action_name: Name of managed action type.
        service: Parent action service reference.
        action_handler_factory: Creates action handlers.
        failure_action_factory: Creates failure frames.
    """

    def __init__(
        self,
        frame_types_to_process: tuple[type[Frame], ...],
        action_name: str,
        service: "BaseActionService",
        action_handler_factory: Callable[[Frame], ActionHandler],
        failure_action_factory: Callable[[str, Frame], ActionFrame],
    ):
        """Initializes modality manager.

        Args:
            frame_types_to_process: Frame types this manager should handle.
            action_name: Name of action type managed by this instance.
            service: Reference to parent action service.
            action_handler_factory: Factory function creating action handlers.
            failure_action_factory: Factory function creating failure action frames.
        """
        self.frame_types_to_process: tuple[type[Frame], ...] = frame_types_to_process
        self.action_name = action_name
        self.service = service
        self.action_handler_factory = action_handler_factory
        self.failure_action_factory = failure_action_factory

    def can_handle_frame(self, frame: Frame) -> bool:
        """Check if the modality manager can handle a given frame."""
        if isinstance(frame, InternalStateMachineTriggerFrame):
            return frame.action_name == self.action_name
        else:
            return isinstance(frame, self.frame_types_to_process)

    async def on_started(self, frame: ActionFrame) -> None:
        """Handle when an action is started."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            logger.warning(f"Unknown action with ID={frame.action_id}. Ignoring {type(frame)} frame {frame}")
            await self.service.queue_for_internal_processing(self.failure_action_factory("Action crashed", frame))
            return
        if not await action_handler.may_started():  # type: ignore
            logger.warning(f"Got frame {frame} when not in 'starting' state")
            logger.warning(f"On action state {action_handler.action_state}")
        await action_handler.started(frame)  # type: ignore

    async def on_stop(self, frame: ActionFrame) -> None:
        """Handle when an action is stopped."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            logger.warning(f"Unknown action with ID={frame.action_id}. Ignoring {type(frame)} frame {frame}")
            await self.service.queue_for_internal_processing(
                self.failure_action_factory("Action does not exist", frame)
            )
            return
        await action_handler.stop(frame)  # type: ignore

    async def on_change(self, frame: ActionFrame) -> None:
        """Handle when an action is changed."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            logger.warning(f"Unknown action with ID={frame.action_id}. Ignoring {type(frame)} frame {frame}")
            await self.service.queue_for_internal_processing(self.failure_action_factory("Action crashed", frame))
            return
        await action_handler.change(frame)  # type: ignore

    async def on_updated(self, frame: ActionFrame) -> None:
        """Handle when an action is updated."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            logger.warning(f"Unknown action with ID={frame.action_id}. Ignoring {type(frame)} frame {frame}")
            await self.service.queue_for_internal_processing(self.failure_action_factory("Action crashed", frame))
            return
        await action_handler.updated(frame)  # type: ignore

    async def on_internal_frame(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle when an internal frame is received."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            action_handler = await self._add_action_handler(frame)
        if not action_handler:
            logger.error(f"Unknown action with ID={frame.action_id}. Ignoring internal {frame.trigger} frame {frame}")
            return

        await action_handler.trigger(frame.trigger, frame)  # type: ignore

    async def on_start(self, frame: ActionFrame) -> None:
        """Handle when an action is starting."""
        if self._get_action_handler(frame):
            logger.warning(f"Action with ID={frame.action_id} already exists. Ignoring {type(frame)} frame {frame}")
            return

        action_handler = await self._add_action_handler(frame)
        await action_handler.start(frame)  # type: ignore
        await self.service.queue_for_internal_processing(
            self._get_internal_frame("check_modality", action_id=frame.action_id)
        )

    async def on_check_modality(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle when a modality check is requested."""
        await self._manage_modality_policy()

    async def on_modality_available(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle modality availability event."""
        action_handler = self._get_action_handler(frame)
        if action_handler.state == "scheduled":  # type: ignore
            await action_handler.trigger(frame.trigger, frame)  # type: ignore

    async def on_finished(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle when an action is finished."""
        action_handler = self._get_action_handler(frame)
        if not action_handler:
            logger.warning(
                f"Action with ID={frame.action_id} not currently running. Ignoring {type(frame)} frame {frame}"
            )
            return

        if action_handler.state != "stopping":  # type: ignore
            logger.warning(f"Got frame {frame} when not in 'stopping' state")
            logger.warning(f"On action state {action_handler.action_state}")

        await action_handler.finished(frame)  # type: ignore
        await self._remove_action_handler(frame.action_id)

    async def _add_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler:
        raise NotImplementedError()

    def _get_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler | None:
        raise NotImplementedError()

    def _get_failure_action_finished(self, reason: str, frame: ActionFrame) -> ActionFrame:
        raise NotImplementedError()

    def _get_internal_frame(
        self, trigger: str, *, action_id: str, data: dict[str, Any] | None = None
    ) -> InternalStateMachineTriggerFrame:
        return InternalStateMachineTriggerFrame(
            trigger, action_id=action_id, action_name=self.action_name, data=data or {}
        )

    async def _remove_action_handler(self, action_id: str) -> None:
        raise NotImplementedError()

    async def _manage_modality_policy(self) -> None:
        """Manage the modality policy.

        Depending on the modality policy, this will do different things.
        """
        raise NotImplementedError()


class BaseActionService(FrameProcessor):
    """Base class for all action services.

    Provides core functionality for processing action frames and managing modalities
    in the system.


    Attributes:
        name: Service name identifier.
        frame_types_to_process: Frame types this service processes.
        modality_managers: List of modality managers.
        stream_id: Current stream identifier.
    """

    name: str
    frame_types_to_process: tuple[type[Frame], ...] = tuple()

    def __init__(self, modality_managers: list[ModalityManager], **kwargs):
        """Initialize the action service.

        Args:
            modality_managers: List of modality managers for different action types.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.modality_managers = modality_managers
        self.stream_id: str | None = None
        self._processing_queue: asyncio.Queue[Frame] = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._should_terminate = False

    async def start(self, frame: StartFrame) -> None:
        """Called during pipeline start."""
        if "stream_id" not in frame.metadata:
            raise ValueError(
                "No 'stream_id' found in frame.metadata. "
                "You need to populate this with PipelineParams(start_metadata={'stream_id': 'id'})"
            )
        self.stream_id = frame.metadata["stream_id"]
        self._processing_task = self.create_task(self.processing_task())

    async def stop(self, frame: EndFrame) -> None:
        """Called during pipeline end."""

    async def cancel(self, frame: CancelFrame) -> None:
        """Called during pipeline cancelled."""

    async def cleanup(self):
        """Called during pipeline cleanup."""
        await super().cleanup()
        self._should_terminate = True
        if self._processing_task:
            await self.cancel_task(self._processing_task)
            self._processing_task = None

    async def queue_for_internal_processing(self, frame: Frame):
        """Queue a frame for internal processing."""
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        await self._processing_queue.put(frame)

    def can_handle_frame(self, frame: Frame, direction: FrameDirection) -> bool:
        """Check if the service can handle a given frame."""
        return (
            isinstance(frame, InternalStateMachineTriggerFrame)
            or isinstance(frame, self.frame_types_to_process)
            and direction == FrameDirection.DOWNSTREAM
        )

    async def processing_task(self):
        """Processing task for the action service."""
        while not self._should_terminate:
            try:
                frame = await self._processing_queue.get()
                for manager in self.modality_managers:
                    if manager.can_handle_frame(frame):
                        if isinstance(frame, StartedActionFrame):
                            await manager.on_started(frame)
                        elif isinstance(frame, StartActionFrame):
                            await manager.on_start(frame)
                        elif isinstance(frame, FinishedActionFrame):
                            await manager.on_finished(frame)
                        elif isinstance(frame, UpdatedActionFrame):
                            await manager.on_updated(frame)
                        elif isinstance(frame, ChangeActionFrame):
                            await manager.on_change(frame)
                        elif isinstance(frame, StopActionFrame):
                            await manager.on_stop(frame)
                        elif isinstance(frame, InternalStateMachineTriggerFrame | InternalStateMachineAbortFrame):
                            if frame.trigger == "check_modality":
                                await manager.on_check_modality(frame)
                            elif frame.trigger == "modality_available":
                                await manager.on_modality_available(frame)
                            else:
                                await manager.on_internal_frame(frame)

                if isinstance(frame, StartedActionFrame | FinishedActionFrame | UpdatedActionFrame):
                    await self.push_frame(frame, FrameDirection.UPSTREAM)
                    await self.push_frame(frame, FrameDirection.DOWNSTREAM)

            except Exception as e:
                logger.exception(f"Error in {self.name} processing task: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame."""
        await super().process_frame(frame, direction)

        # Internal frames are not propagated all other frames are immediately pushed
        if isinstance(frame, InternalStateMachineTriggerFrame):
            logger.error(f"Received internal frame {frame}")
            return

        await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
        elif self.can_handle_frame(frame, direction):
            await self.queue_for_internal_processing(frame)


class OverrideModalityManager(ModalityManager):
    """Manages actions using override policy.

    The override policy implements an action stack. Multiple actions can run at the same time.
    If an action is already running when a new action should be started, the current
    action will be paused ("overwritten"). When the overwriting action finishes,
    the other action will be resumed.

    Attributes:
        action_stack: Stack of currently active actions.
    """

    def __init__(
        self,
        frame_types_to_process: tuple[type[Frame], ...],
        action_name: str,
        service: "BaseActionService",
        action_handler_factory: Callable[[Frame], ActionHandler],
        failure_action_factory: Callable[[str, Frame], ActionFrame],
    ):
        """Initialize the override modality manager.

        Args:
            frame_types_to_process: Tuple of frame types this manager should handle.
            action_name: Name of the action type managed by this instance.
            service: Reference to the parent action service.
            action_handler_factory: Factory function to create action handlers.
            failure_action_factory: Factory function to create failure action frames.
        """
        super().__init__(
            frame_types_to_process,
            action_name,
            service,
            action_handler_factory,
            failure_action_factory,
        )
        self.action_stack: list[ActionHandler] = []

    @property
    def focus_action_index(self) -> int | None:
        """Get the index of the focus action."""
        if len(self.action_stack) == 0:
            return None

        for i, a in enumerate(self.action_stack):
            if a.state == "scheduled":  # type: ignore
                return i - 1 if i > 0 else None

        return len(self.action_stack) - 1

    def _get_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler | None:
        """Get the action handler for a given frame."""
        for action in reversed(self.action_stack):
            if action.action_id == frame.action_id:
                return action

        return None

    async def on_internal_frame(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle when an internal frame is received."""
        if frame.trigger == "resume" and self.action_stack[-1].action_id != frame.action_id:
            logger.info(
                f"Cannot resume Action({short_id(frame.action_id)}) because "
                "it is no longer the topmost action on the stack."
            )
            return

        if frame.trigger == "pause" and self._get_action_handler(frame).state == "paused":
            # If the action is already paused, we don't need to do anything
            return

        await super().on_internal_frame(frame)

    async def on_modality_available(self, frame: InternalStateMachineTriggerFrame) -> None:
        """Handle modality availability event."""
        for i, action in enumerate(self.action_stack):
            if action.action_id == frame.action_id:
                if self.focus_action_index and i == self.focus_action_index:
                    # If the action is already in focus, we don't need to do anything
                    return
                elif self.focus_action_index and i - self.focus_action_index != 1:
                    logger.warning(
                        "Focus action and next focus action differ in index != 1. "
                        f"Difference: {i - self.focus_action_index}"
                    )
                break

        await super().on_modality_available(frame)

    async def _manage_modality_policy(self) -> None:
        """Manage the modality policy.

        Manages the modality policy for the action stack. Actions can be scheduled (s), starting (t), paused (p),
        running (r) or finished (f). In general the stack at any point can look like this:
        [p, ... , p | r | t , s, s, ...]
        """
        if len(self.action_stack) == 0:
            return

        if self.focus_action_index is None:
            await self.service.queue_for_internal_processing(
                self._get_internal_frame("modality_available", action_id=self.action_stack[0].action_id)
            )
        else:
            focus_action = self.action_stack[self.focus_action_index]
            if self.focus_action_index < len(self.action_stack) - 1:
                action_state = focus_action.state  # type: ignore
                short_action_id = short_id(focus_action.action_id)
                next_action_id = self.action_stack[self.focus_action_index + 1].action_id

                logger.info(f"Modality check: Focused Action({short_action_id}) is in state '{action_state}'")
                if action_state == "paused" or action_state == "finished":
                    await self.service.queue_for_internal_processing(
                        self._get_internal_frame("modality_available", action_id=next_action_id)
                    )

                elif action_state == "running":
                    await self.service.queue_for_internal_processing(
                        self._get_internal_frame("pause", action_id=focus_action.action_id)
                    )

                # Allow for other tasks to run
                await asyncio.sleep(0)
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame("check_modality", action_id=next_action_id)
                )

    async def _add_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler:
        action_handler = self.action_handler_factory(frame)
        self.action_stack.append(action_handler)
        return action_handler

    async def _remove_action_handler(self, action_id: str) -> None:
        top_action: ActionHandler | None = self.action_stack[-1] if len(self.action_stack) > 0 else None
        action_to_remove: ActionHandler | None = None

        for action in reversed(self.action_stack):
            if action.action_id == action_id:
                action_to_remove = action

        if action_to_remove:
            self.action_stack.remove(action_to_remove)

            if len(self.action_stack) == 0:
                await action_to_remove.clear_modality()
            elif action_to_remove == top_action:
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame(
                        "resume",
                        action_id=self.action_stack[-1].action_id,
                        data={"previous_action_state": action_to_remove.action_state},
                    )
                )


class ParallelModalityManager(ModalityManager):
    """Manages actions using parallel policy.

    Allows multiple actions to run simultaneously without interference.

    Attributes:
        actions: Dictionary mapping action IDs to handlers.
    """

    def __init__(
        self,
        frame_types_to_process: tuple[type[Frame], ...],
        action_name: str,
        service: "BaseActionService",
        action_handler_factory: Callable[[Frame], ActionHandler],
        failure_action_factory: Callable[[str, Frame], ActionFrame],
    ):
        """Initialize the parallel modality manager.

        Args:
            frame_types_to_process: Tuple of frame types this manager should handle.
            action_name: Name of the action type managed by this instance.
            service: Reference to the parent action service.
            action_handler_factory: Factory function to create action handlers.
            failure_action_factory: Factory function to create failure action frames.
        """
        super().__init__(
            frame_types_to_process,
            action_name,
            service,
            action_handler_factory,
            failure_action_factory,
        )
        self.actions: dict[str, ActionHandler] = {}

    def _get_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler | None:
        return self.actions.get(frame.action_id, None)

    async def _manage_modality_policy(self) -> None:
        for action in self.actions.values():
            if action.state == "scheduled":  # type: ignore
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame("modality_available", action_id=action.action_id)
                )

    async def _add_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler:
        self.actions[frame.action_id] = self.action_handler_factory(frame)
        return self.actions[frame.action_id]

    async def _remove_action_handler(self, action_id: str) -> None:
        if action_id in self.actions:
            del self.actions[action_id]


class ReplaceModalityManager(ModalityManager):
    """Manages actions using replace policy.

    If a new action should be started, any ongoing action is stopped and the new action
    is started after that.

    Attributes:
        action: Currently active action handler.
        replaced_actions: Queue of actions that have been replaced.
    """

    def __init__(
        self,
        frame_types_to_process: tuple[type[Frame], ...],
        action_name: str,
        service: "BaseActionService",
        action_handler_factory: Callable[[Frame], ActionHandler],
        failure_action_factory: Callable[[str, Frame], ActionFrame],
    ):
        """Initialize the replace modality manager.

        Args:
            frame_types_to_process: Tuple of frame types this manager should handle.
            action_name: Name of the action type managed by this instance.
            service: Reference to the parent action service.
            action_handler_factory: Factory function to create action handlers.
            failure_action_factory: Factory function to create failure action frames.
        """
        super().__init__(
            frame_types_to_process,
            action_name,
            service,
            action_handler_factory,
            failure_action_factory,
        )
        self.action: ActionHandler | None = None
        self.replaced_actions: deque[ActionHandler] = deque()

    def _get_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler | None:
        # If action_id matches current action
        if self.action and self.action.action_id == frame.action_id:
            return self.action

        # otherwise check all replaced actions...
        for replaced_action in self.replaced_actions:
            if replaced_action and replaced_action.action_id == frame.action_id:
                return replaced_action

        # ... or return None if no match was found
        return None

    async def _manage_modality_policy(self) -> None:
        if self.action is None:
            return
        if len(self.replaced_actions) > 0:
            newest_action_to_be_replaced = self.replaced_actions[-1]
            replaced_action_state: str = newest_action_to_be_replaced.state  # type: ignore
            if replaced_action_state == "finished":
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame("modality_available", action_id=self.action.action_id)
                )
            elif replaced_action_state == "stopping":
                # Allow for other tasks to run
                await asyncio.sleep(0)
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame("check_modality", action_id=self.action.action_id)
                )
            else:
                await self.service.queue_for_internal_processing(
                    InternalStateMachineAbortFrame(
                        action_name=self.action_name,
                        action_id=newest_action_to_be_replaced.action_id,
                        reason="Action replaced.",
                    )
                )
                # Allow for other tasks to run
                await asyncio.sleep(0)
                await self.service.queue_for_internal_processing(
                    self._get_internal_frame("check_modality", action_id=self.action.action_id)
                )

        else:
            await self.service.queue_for_internal_processing(
                self._get_internal_frame("modality_available", action_id=self.action.action_id)
            )

    async def _add_action_handler(self, frame: ActionFrame | InternalStateMachineTriggerFrame) -> ActionHandler:
        if self.action is not None:
            self.replaced_actions.append(self.action)
        self.action = self.action_handler_factory(frame)
        return self.action

    async def _remove_action_handler(self, action_id: str) -> None:
        if self.action and action_id == self.action.action_id:
            await self.action.clear_modality()
            self.action = None
        else:
            to_remove = None
            for replaced_action in self.replaced_actions:
                if replaced_action and action_id == replaced_action.action_id:
                    to_remove = replaced_action
            if to_remove:
                self.replaced_actions.remove(to_remove)
