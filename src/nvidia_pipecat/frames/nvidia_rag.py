# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA RAG frames."""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, ServiceUpdateSettingsFrame
from pydantic import BaseModel


@dataclass
class NvidiaRAGSettingsFrame(ServiceUpdateSettingsFrame):
    """A frame to update the settings for NvidiaRAG."""


class NvidiaRAGCitation(BaseModel):
    """A model class to contain NvidiaRAG's citation data.

    Args:
        document_type: Type of document (text, chart, etc.).
        document_id: ID of the document.
        document_name: Name of the document.
        content: Content of citation as a base64 image.
        metadata: Metadata of citation (language, date created, last modified, etc.).
        score: Score from the ranking model.
    """

    document_type: str
    document_id: str
    document_name: str
    content: bytes
    metadata: str
    score: float


@dataclass
class NvidiaRAGCitationsFrame(DataFrame):
    """A frame that contains NvidiaRAG's citations.

    Args:
        citations: List of citations, each being a NvidiaRAGCitation object.
    """

    citations: list[NvidiaRAGCitation]
