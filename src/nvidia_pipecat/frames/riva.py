# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Riva frames for Interim Transcription.

This module provides frame definitions for NVIDIA Riva's speech-to-text functionality,
specifically focused on interim transcription handling.

Classes:
    RivaInterimTranscriptionFrame: Frame for interim transcription results with stability metrics
"""

from dataclasses import dataclass

from pipecat.frames.frames import InterimTranscriptionFrame


@dataclass
class RivaInterimTranscriptionFrame(InterimTranscriptionFrame):
    """An interim transcription frame with stability metrics from Riva.

    Extends the base InterimTranscriptionFrame to include Riva-specific stability
    scoring for speculative speech processing. These frames are generated during
    active speech and help determine when to trigger early response generation.

    Also see:
    - InterimTranscriptionFrame : Base class for interim transcriptions

    Args:
        stability (float): Confidence score for the transcription, ranging 0.0-1.0.
            - 0.0: Highly unstable, likely to change
            - 1.0: Maximum stability, no expected changes
            Only transcripts with stability=1.0 are processed for speculative
            speech handling. Defaults to 0.1.
        user_id (str): Identifier of the speaking participant.
        text (str): The interim transcription text.
        language (str): Language code of the transcription.
        timestamp (float): Timestamp of when the transcription was generated.

    Typical usage example:
        >>> frame = RivaInterimTranscriptionFrame(
        ...     text="Hello world",
        ...     stability=0.95,
        ...     user_id="user_1",
        ...     language="en-US",
        ...     timestamp=1234567890.0
        ... )
        >>> print(frame)  # Output will be:
        RivaInterimTranscriptionFrame(
            user: user_1,
            text: [Hello world],
            stability: 0.95,
            language: en-US,
            timestamp: 1234567890.0
        )
    """

    stability: float = 0.1

    def __str__(self):
        """Return a string representation of the frame.

        Returns:
            str: A formatted string containing all frame attributes.
        """
        return (
            f"{self.name}(user: {self.user_id}, text: [{self.text}], "
            f"stability: {self.stability}, language: {self.language}, timestamp: {self.timestamp})"
        )
