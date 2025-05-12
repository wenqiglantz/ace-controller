# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""ElevenLabs specific frames.

This module provides frames specific to the ElevenLabs services.
"""

from typing import Any, Optional, Union

from pipecat.frames.frames import TranscriptionFrame


class ElevenLabsInterimTranscriptionFrame(TranscriptionFrame):
    """ElevenLabs interim transcription frame.

    This frame represents an interim transcription result from ElevenLabs ASR.
    It extends the TranscriptionFrame to include a stability score.

    Attributes:
        text: The transcribed text.
        language: ISO language code.
        timestamp: ISO 8601 timestamp when the transcription was generated.
        metadata: Optional additional metadata.
        stability: Confidence score from 0.0 to 1.0.
    """

    def __init__(
        self,
        text: str,
        language: str,
        timestamp: str,
        metadata: Optional[dict[str, Any]] = None,
        stability: float = 0.0,
    ):
        """Initialize an ElevenLabsInterimTranscriptionFrame.

        Args:
            text: The transcribed text.
            language: ISO language code.
            timestamp: ISO 8601 timestamp when the transcription was generated.
            metadata: Optional additional metadata.
            stability: Confidence score from 0.0 to 1.0.
        """
        super().__init__(text, language, timestamp, metadata)
        self.stability = stability 