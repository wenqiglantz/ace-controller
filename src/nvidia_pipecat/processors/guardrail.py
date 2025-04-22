# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""This module defines the `GuardrailProcessor` for handling blocked words/topics in user queries.

Ensures that specific queries can be detected and blocked from further processing.
"""

import re

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GuardrailProcessor(FrameProcessor):
    """Blocks queries containing specified words or topics.

    Args:
        blocked_words (list[str]): List of words that trigger query blocking.
            Matching is case-insensitive and uses word boundaries.
        block_message (str): Message returned when a query is blocked.
            Defaults to "I am not allowed to answer this question".
        **kwargs: Additional arguments passed to parent FrameProcessor.
    """

    def __init__(self, blocked_words=None, block_message="I am not allowed to answer this question", **kwargs):
        """Initializes the GuardrailProcessor.

        Args:
            blocked_words (list[str] or None): A list of words for which queries should be blocked.
            block_message (str): The message to return when a blocked query is detected.
            **kwargs: Additional keyword arguments passed to the parent class initializer.
        """
        super().__init__(**kwargs)
        # Normalize all words to lowercase for case-insensitive matching
        self._blocked_words = [word.lower() for word in (blocked_words or [])]
        self._block_message = block_message

    def is_query_blocked(self, query: str) -> bool:
        """Checks if the given query contains any blocked word.

        Args:
            query (str): The query text to evaluate.

        Returns:
            bool: True if query contains any blocked word, False otherwise.
        """
        query_lower = query.lower()  # Convert query to lowercase for case-insensitive matching
        for word in self._blocked_words:
            # Use regex to match the word as a whole word (word boundaries)
            if re.search(rf"\b{re.escape(word)}\b", query_lower):
                logger.debug(f"Query: {query} contains blocked word: {word}")
                return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames and blocks those containing restricted content.

        Args:
            frame (Frame): The incoming frame to process.
            direction (FrameDirection): The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        # Handle user queries
        if isinstance(frame, TranscriptionFrame) and self.is_query_blocked(frame.text):
            logger.debug(f"Blocked query detected: {frame.text}")
            # Respond with blocking message
            await self.push_frame(TTSSpeakFrame(self._block_message))
            return  # Stop further propagation of the blocked query

        # For all other frames, proceed as usual
        await super().push_frame(frame, direction)
