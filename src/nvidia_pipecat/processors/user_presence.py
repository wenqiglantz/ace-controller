# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""User presence detection and greeting processor for ACE conversation pipeline."""

from loguru import logger
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    InputAudioRawFrame,
    StartInterruptionFrame,
    SystemFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.action import FinishedPresenceUserActionFrame, StartedPresenceUserActionFrame


class UserPresenceProcesssor(FrameProcessor):
    """Manages user presence detection and automated greetings in the ACE conversation pipeline.

    Manages automated greetings and pipeline activation based on user presence state.
    Sends welcome/farewell messages and controls frame forwarding accordingly.

    Input Frames:
        StartedPresenceUserActionFrame: User becomes present
        FinishedPresenceUserActionFrame: User leaves
        UserStartedSpeakingFrame: User speech event
        InputAudioRawFrame: Raw audio input
        SystemFrame: System events
        ControlFrame: Control events
    """

    def __init__(self, welcome_msg="Hello", farewell_msg="Goodbye", **kwargs):
        """Initialize the user presence processor.

        Args:
            welcome_msg (str): Message spoken when user presence detected.
            farewell_msg (str): Message spoken when user leaves.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._welcome_msg = welcome_msg
        self._farewell_msg = farewell_msg
        self._is_user_present = False  # At startup, it is assumed that the user is not present

    async def _greet_welcome(self):
        """Internal method to handle welcome greeting."""
        logger.debug("User detected. Greeting a welcome")
        try:
            await self.push_frame(TTSSpeakFrame(self._welcome_msg))
            self._is_user_present = True
        except Exception as e:
            logger.error(e)

    async def _greet_farewell(self):
        """Internal method to handle farewell greeting."""
        logger.debug(f"User left. Bidding farewell. {self._farewell_msg}")
        try:
            await self.push_frame(TTSSpeakFrame(self._farewell_msg))
            self._is_user_present = False
        except Exception as e:
            logger.error(e)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and manage user presence state.

        Handles user presence detection, greeting messages, and controls frame forwarding
        based on user presence state.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartedPresenceUserActionFrame):
            # Welcome the user on detecting presence
            await self.push_frame(frame, direction)
            await self._greet_welcome()

        elif isinstance(frame, FinishedPresenceUserActionFrame):
            # Greet farewell to the user on detecting absence.
            await self.push_frame(StartInterruptionFrame())
            await self.push_frame(frame, direction)
            await self._greet_farewell()

        elif isinstance(frame, SystemFrame | ControlFrame):
            if isinstance(frame, UserStartedSpeakingFrame | InputAudioRawFrame) and not self._is_user_present:
                return
            else:
                await self.push_frame(frame, direction)
        else:
            # Pass all frames if user is present
            if self._is_user_present:
                await self.push_frame(frame, direction)
            else:
                logger.debug(f"Frame {frame} blocked as no user is present")
