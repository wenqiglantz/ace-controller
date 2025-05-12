#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""
Example speech-to-speech pipeline using ElevenLabs ASR and TTS services.

This example shows how to create a voice conversation pipeline using:
- ACETransport for audio input/output
- ElevenLabsASRService for speech recognition
- NvidiaLLMService for text generation
- ElevenLabsTTSService for speech synthesis

Note that this requires valid API keys for ElevenLabs and NVIDIA services.
"""

import asyncio
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero_vad_analyzer import SileroVADAnalyzer
from pipecat.frames.frames import StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.ace.config import ACETransportParams
from pipecat.transports.ace.transport import ACETransport

from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services import ElevenLabsASRService, ElevenLabsTTSServiceWithEndOfSpeech, NvidiaLLMService
from nvidia_pipecat.transports.services.ace_controller.http_server import HTTPServer
from nvidia_pipecat.utils.logging import setup_default_ace_logging

# Set up logging
setup_default_ace_logging(level="DEBUG")

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("ELEVENLABS_API_KEY"):
    logger.error("ELEVENLABS_API_KEY environment variable not found. This is required for ElevenLabs services.")
    exit(1)

if not os.getenv("NVIDIA_API_KEY"):
    logger.error("NVIDIA_API_KEY environment variable not found. This is required for NVIDIA LLM service.")
    exit(1)


class PipelineMetadata:
    """Metadata for the pipeline task."""

    def __init__(self, websocket):
        """Initialize pipeline metadata.

        Args:
            websocket: WebSocket connection for the pipeline.
        """
        self.websocket = websocket


async def create_pipeline_task(pipeline_metadata: PipelineMetadata):
    """Create the pipeline to be run.

    Args:
        pipeline_metadata (PipelineMetadata): Metadata containing websocket and other pipeline configuration.

    Returns:
        PipelineTask: The configured pipeline task for handling speech-to-speech conversation.
    """
    transport = ACETransport(
        websocket=pipeline_metadata.websocket,
        params=ACETransportParams(
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        model="meta/llama-3.1-8b-instruct",
        base_url=None,
    )

    # Use ElevenLabs ASR service instead of Riva ASR
    stt = ElevenLabsASRService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        model="scribe_v1",
        language="en-US",
        sample_rate=16000,
        diarize=False,
        tag_audio_events=True,
        timestamps_granularity="word",
        chunk_size_seconds=3,  # Buffer audio in 3-second chunks
    )
    
    # Use ElevenLabs TTS service
    tts = ElevenLabsTTSServiceWithEndOfSpeech(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),  # Default to "Rachel"
        sample_rate=16000,
        model="eleven_turbo_v2",
    )
    
    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always answer as helpful friendly and polite. "
            "Respond with one sentence or less than 75 characters. Do not respond with bulleted or numbered list. "
            "Your output will be converted to audio so don't include special characters in your answers.",
        },
    ]

    # Create and run the pipeline
    pipeline = Pipeline(
        "speech-to-speech-pipeline-with-elevenlabs",
        [
            transport,
            stt,
            stt_transcript_synchronization,
            llm,
            tts_transcript_synchronization,
            tts,
            transport,
        ],
    )

    start_frame = StartFrame(
        metadata={
            "stream_id": pipeline_metadata.websocket.get_extra_info("websocket_id"),
            "messages": messages,
        }
    )

    task = PipelineTask(pipeline=pipeline, first_frame=start_frame)
    return task


async def start_server():
    """Start the HTTP server with the ACE Controller."""
    static_root = str(Path(__file__).parent.parent / "static")
    logger.info(f"Starting server with static files at {static_root}")
    
    server = HTTPServer(
        static_root=static_root,
        create_pipeline_task=create_pipeline_task,
        pipeline_metadata_cls=PipelineMetadata,
    )
    
    await server.start()
    
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )
    
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    logger.info("Starting ElevenLabs ASR/TTS example server...")
    asyncio.run(start_server()) 