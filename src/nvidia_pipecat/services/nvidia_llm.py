# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""NVIDIA LLM service for connecting to NVIDIA LLM endpoints.

Provides interface to NVIDIA's language model endpoints with support for streaming
responses and parameter customization.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages.base import BaseMessageChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    TextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
)
from pydantic import BaseModel, Field

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced


@traceable
class NvidiaLLMService(LLMService):
    """Service for interacting with NVIDIA language models.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages

    Attributes:
        model_name: Identifier of the language model being used.
    """

    class InputParams(BaseModel):
        """Parameters for controlling NVIDIA LLM behavior."""

        frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
        presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
        seed: int | None = Field(default=None, ge=0)
        temperature: float | None = Field(default=None, ge=0.0, le=2.0)
        top_k: int | None = Field(default=None, ge=0)
        top_p: float | None = Field(default=None, ge=0.0, le=1.0)
        max_tokens: int | None = Field(default=None, ge=1)
        max_completion_tokens: int | None = Field(default=None, ge=1)
        extra: dict[str, Any] | None = Field(default=None)

    def __init__(
        self,
        *,
        model: str = "meta/llama3-8b-instruct",
        api_key: str = None,
        base_url: str | None = None,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initializes NVIDIA LLM service.

        Args:
            model: Model identifier for chat completions.
            api_key: Authentication key for NVIDIA AI endpoints.
            base_url: Base URL for NVIDIA AI endpoints. Defaults to NVIDIA cloud if None.
            params: Model behavior parameters. Uses defaults if None.
            **kwargs: Additional arguments passed to parent LLMService.

        Note:
            If params is not provided, default values from InputParams will be used.
            The service supports various parameters like temperature, top_p, etc.,
            which can be configured through the InputParams class.

        Usage:
            If base_url is not set then it defaults to "https://api.nvidia.com/v1" and use NVIDIA AI endpoints.
            API key is required for NVIDIA AI endpoints.
            For locally deploying NIM LLMs refer to https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html,
            and set base_url to the local endpoint.
        """
        super().__init__(**kwargs)
        if params is None:
            params = NvidiaLLMService.InputParams()
        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self.set_model_name(model)
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> OpenAIContextAggregatorPair:
        """Creates context aggregator pair.

        Args:
            context: OpenAI LLM context to aggregate.
            assistant_expect_stripped_words: Whether assistant expects stripped words.

        Returns:
            Pair of user and assistant context aggregators.
        """
        user = OpenAIUserContextAggregator(context)
        assistant = OpenAIAssistantContextAggregator(context, expect_stripped_words=assistant_expect_stripped_words)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create a client for the NVIDIA LLM service."""
        return ChatNVIDIA(
            base_url=base_url,
            model=self.model_name,
            api_key=api_key,
        )

    def can_generate_metrics(self) -> bool:
        """Check if the NVIDIA LLM service can generate metrics."""
        return True

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="llm")
    async def get_chat_completions(self, context: OpenAILLMContext, messages) -> AsyncIterator[BaseMessageChunk]:
        """Gets streaming chat completions from model.

        Args:
            context: Context containing conversation history and settings.
            messages: List of conversation messages.

        Returns:
            Iterator yielding response chunks from model.
        """
        params = {
            "model": self.model_name,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }
        params.update(self._settings["extra"])
        chunks = self._client.astream(input=messages, config=params)
        return chunks

    async def _stream_chat_completions(self, context: OpenAILLMContext) -> AsyncIterator[BaseMessageChunk]:
        """Streams chat completions from model.

        Args:
            context: Context containing conversation history and settings.

        Returns:
            Iterator yielding response chunks from model.
        """
        logger.debug(f"Generating chat: {context.get_messages_for_logging()}")
        messages = context.get_messages()
        chunks = await self.get_chat_completions(context, messages)
        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        """Process the context for the NVIDIA LLM service."""
        await self.start_ttfb_metrics()
        chunk_stream: AsyncIterator[BaseMessageChunk] = await self._stream_chat_completions(context)
        try:
            async for chunk in chunk_stream:
                if not chunk.content:
                    continue
                await self.stop_ttfb_metrics()
                await self.push_frame(TextFrame(chunk.content))
        except asyncio.CancelledError:
            logger.debug("Task _stream_chat_completions was cancelled!")
            raise

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes incoming frames.

        Handles context frames, message frames, and setting updates.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)
        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)
        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
