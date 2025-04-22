# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Acknowledgment processor that provides verbal feedback during conversation pauses.

A simple processor that adds natural conversational acknowledgments when users stops speaking if the LLM or RAG
is taking a long time to respond, helping to create more engaging interactions.
"""

import random

from loguru import logger
from pipecat.frames.frames import Frame, TTSSpeakFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AcknowledgmentProcessor(FrameProcessor):
    """Processor that sends configurable acknowledgment responses during conversation pauses.

    This processor enhances conversation flow by sending occasional acknowledgment words
    (like "Hmmm" or "Let me think") when users pause speaking. It only works when 2-phase
    End-of-Utterance (EOU) detection is disabled.

    Input Frames:
        - UserStoppedSpeakingFrame (consumed): Indicates when a user has stopped speaking

    Output Frames:
        - TTSSpeakFrame: Contains the acknowledgment text to be spoken

    Args:
        filler_words (list[str]): List of acknowledgment phrases to use.
            Each phrase should be a short, natural acknowledgment (e.g., "Hmmm", "Let me think").
        filler_probability (float): Probability (0.0 to 1.0) of sending an
            acknowledgment when a pause is detected. Defaults to 0.5.
    """

    def __init__(self, filler_words=None, filler_probability=0.5, **kwargs):
        """Initialize the acknowledgment processor.

        Args:
            filler_words (list[str]): List of acknowledgment phrases to use.
                Each phrase should be a short, natural acknowledgment (e.g., "Hmmm", "Let me think").
            filler_probability (float, optional): Probability (0.0 to 1.0) of sending an
                acknowledgment when a pause is detected. Defaults to 0.5.
            **kwargs: Additional arguments passed to the parent FrameProcessor.

        Raises:
            ValueError: If filler_probability is not between 0.0 and 1.0
            ValueError: If filler_words is empty or None
        """
        super().__init__(**kwargs)
        self.filler_words = filler_words
        self.filler_probability = filler_probability

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and generate acknowledgments when appropriate.

        Args:
            frame (Frame): The incoming frame to process
            direction (FrameDirection): The direction the frame is traveling

        Returns:
            None
        """
        await super().process_frame(frame, direction)

        # If user is present and they've just stopped speaking, send a filler TTSSpeakFrame
        if isinstance(frame, UserStoppedSpeakingFrame):
            # Add a probability to skip sending a filler word
            filler_probability = self.filler_probability
            if random.random() < filler_probability:
                filler = random.choice(self.filler_words)
                logger.debug(f"User stopped speaking, sending filler word: {filler}")
                filler_frame = TTSSpeakFrame(filler)
                await self.push_frame(filler_frame, FrameDirection.DOWNSTREAM)
            else:
                filler = ""
                logger.debug(f"User stopped speaking, sending filler word: {filler}")
                filler_frame = TTSSpeakFrame(filler)
                await self.push_frame(filler_frame, FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)
