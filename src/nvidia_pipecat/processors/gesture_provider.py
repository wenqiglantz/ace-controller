# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Gesture provider processor.

A frame processor that automatically manages facial expressions for the ACE avatar
based on conversation events and speaking states. Helps create more natural interactions
by adding contextual facial gestures during conversations.

For available facial gestures, see the ACE Animgraph documentation:
https://docs.nvidia.com/ace/animation-graph-microservice/latest/default-animation-graph.html
"""

import random

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartInterruptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.action import StartFacialGestureBotActionFrame


class FacialGestureProviderProcessor(FrameProcessor):
    """Manages automated facial gestures for the ACE avatar during conversations.

    This processor monitors conversation state changes and triggers appropriate facial
    expressions in response to events like the user finishing speaking or interruptions
    occurring. It includes configurable randomization to make gestures feel natural.

    Input Frames:
        - UserStoppedSpeakingFrame (consumed): Triggered when user finishes speaking
        - StartInterruptionFrame (consumed): Triggered during conversation interruptions
        - BotStartedSpeakingFrame (consumed): Indicates bot began speaking
        - BotStoppedSpeakingFrame (consumed): Indicates bot finished speaking

    Output Frames:
        - StartFacialGestureBotActionFrame: Triggers facial expressions on the avatar

    Args:
        user_stopped_speaking_gesture (str): Facial gesture to trigger when user stops speaking.
            See ACE Animgraph docs for available gestures. Defaults to "Taunt".
        start_interruption_gesture (str): Facial gesture to trigger during interruptions.
            See ACE Animgraph docs for available gestures. Defaults to "Pensive".
        probability (float): Probability (0.0 to 1.0) that a gesture will be triggered
            for any given event. Used to make behavior less predictable. Defaults to 0.5.
        **kwargs: Additional arguments passed to parent FrameProcessor.

    Typical usage example:
        >>> processor = FacialGestureProviderProcessor(
        ...     user_stopped_speaking_gesture="Smile",
        ...     start_interruption_gesture="Concerned",
        ...     probability=0.75
        ... )
    """

    def __init__(
        self, user_stopped_speaking_gesture="Taunt", start_interruption_gesture="Pensive", probability=0.5, **kwargs
    ):
        """Initialize the facial gesture provider.

        Args:
            user_stopped_speaking_gesture (str): Facial gesture to trigger when user stops speaking.
                See ACE Animgraph docs for available gestures. Defaults to "Taunt" by default.
            start_interruption_gesture (str): Facial gesture to trigger during interruptions.
                See ACE Animgraph docs for available gestures. Defaults to "Pensive" by default.
            probability (float): Probability (0.0 to 1.0) that a gesture will be triggered
                for any given event. Used to make behavior less predictable. Defaults to 0.5.
            **kwargs: Additional arguments passed to parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self.user_stopped_speaking_gesture = user_stopped_speaking_gesture
        self.start_interruption_gesture = start_interruption_gesture
        self._bot_speaking = False
        self.probability = probability

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame and trigger facial gestures if appropriate.

        Monitors conversation state changes and randomly triggers configured facial
        gestures based on the probability setting.

        Args:
            frame (Frame): The incoming frame to process.
            direction (FrameDirection): The direction the frame is traveling.

        Returns:
            None
        """
        await super().process_frame(frame, direction)

        new_frame: Frame | None = None
        frame_direction: FrameDirection | None = None

        if isinstance(frame, UserStoppedSpeakingFrame):
            if random.random() < self.probability:
                logger.info("User stopped speaking gesture provider")
                new_frame = StartFacialGestureBotActionFrame(facial_gesture=self.user_stopped_speaking_gesture)
                frame_direction = FrameDirection.DOWNSTREAM
        elif isinstance(frame, StartInterruptionFrame):
            logger.info("Start interruption frame gesture provider")
            if self._bot_speaking and random.random() < self.probability:
                new_frame = StartFacialGestureBotActionFrame(facial_gesture=self.start_interruption_gesture)
                frame_direction = FrameDirection.DOWNSTREAM
            self._bot_speaking = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        # Push facial gesture frame after the incoming frame.
        # With this the StartInterruptionFrame will not delete it by resetting the frame queues.
        await self.push_frame(frame, direction)
        await self.push_frame(new_frame, frame_direction)
