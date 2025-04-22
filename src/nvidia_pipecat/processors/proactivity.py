# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Proactivity processor that manages automated bot responses during conversation lulls.

Monitors conversation activity and triggers automated responses after periods of silence
to maintain user engagement.
"""

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.action import FinishedPresenceUserActionFrame, StartedPresenceUserActionFrame


class ProactivityProcessor(FrameProcessor):
    """Manages automated bot responses during conversation pauses.

    Monitors user presence and conversation activity, automatically generating
    proactive messages during extended periods of silence.

    Attributes:
        default_message (str): Message sent when the inactivity timer expires.
        timer_duration (float): Seconds to wait before triggering proactive message.

    Input Frames:
        StartedPresenceUserActionFrame: User becomes present
        FinishedPresenceUserActionFrame: User leaves
        BotStartedSpeakingFrame: Bot starts speaking
        BotStoppedSpeakingFrame: Bot stops speaking
        UserStartedSpeakingFrame: User starts speaking
        UserStoppedSpeakingFrame: User stops speaking
        EndFrame: Pipeline ends
    """

    def __init__(self, default_message: str = "I'm here if you need me!", timer_duration: float = 100, **kwargs):
        """Initializes the processor with specified message and timer settings.

        Args:
            default_message (str): Message sent when inactivity timer expires.
            timer_duration (float): Seconds to wait before sending message.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.default_message = default_message
        self.timer_duration = timer_duration
        self._timer_task = None
        self._user_present = False

    async def _start_timer(self):
        """Start the proactivity timer.

        Internal method that manages the timer countdown and sends the default message
        when the timer expires.
        """
        try:
            logger.debug("Timer started")
            await asyncio.sleep(self.timer_duration)
            logger.info(f"Timer expired, sending default message: {self.default_message}")
            await self.push_frame(TTSSpeakFrame(self.default_message), FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            # Timer cancelled
            logger.debug("Timer cancelled")
            raise

    async def _reset_timer(self):
        """Reset the proactivity timer.

        Internal method that cancels any existing timer and starts a new countdown.
        """
        if self._timer_task:
            await self.cancel_task(self._timer_task)
        logger.debug("Resetting Timer")
        self._timer_task = self.create_task(self._start_timer())

    async def _stop_timer(self):
        """Stop the proactivity timer.

        Internal method that cancels the current timer without starting a new one.
        """
        logger.debug("Stopping timer")
        if self._timer_task:
            await self.cancel_task(self._timer_task)
            self._timer_task = None

    async def cleanup(self):
        """Clean up processor resources.

        Ensures the proactivity timer is properly stopped when the processor shuts down.
        """
        await super().cleanup()
        await self._stop_timer()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage proactivity timer state.

        Handles various conversation events to start, stop, or reset the proactivity timer
        based on user presence and speaking states.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)
        await super().push_frame(frame, direction)

        if isinstance(frame, StartedPresenceUserActionFrame):
            self._user_present = True
            await self._reset_timer()
        elif isinstance(frame, BotStoppedSpeakingFrame | UserStoppedSpeakingFrame):
            # Whenever the user or bot is done talking we want to reset the timer
            if self._user_present:
                await self._reset_timer()
        elif isinstance(frame, BotStartedSpeakingFrame | UserStartedSpeakingFrame):
            # When either the user or the bot starts speaking we don't want to interrupt
            if self._user_present:
                await self._stop_timer()
        elif isinstance(frame, EndFrame):
            # Stop the timer when the pipeline ends
            await self._stop_timer()
        elif isinstance(frame, FinishedPresenceUserActionFrame):
            self._user_present = False
            await self._stop_timer()
