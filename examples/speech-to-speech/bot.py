# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Speech-to-speech conversation bot."""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame

# Uncomment the following line if you want to use ElevenLabsTTS
# from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata

# Uncomment the below lines enable speculative speech processing
# from nvidia_pipecat.processors.nvidia_context_aggregator import (
#     NvidiaTTSResponseCacher,
#     create_nvidia_context_aggregator,
# )
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService
from nvidia_pipecat.transports.network.ace_fastapi_websocket import ACETransport, ACETransportParams
from nvidia_pipecat.transports.services.ace_controller.routers.websocket_router import router as websocket_router
from nvidia_pipecat.utils.logging import setup_default_ace_logging

load_dotenv(override=True)

setup_default_ace_logging(level="DEBUG")


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

    stt = RivaASRService(
        server="localhost:50051",
        api_key=os.getenv("NVIDIA_API_KEY"),
        language="en-US",
        sample_rate=16000,
        model="parakeet-1.1b-en-US-asr-streaming-silero-vad-asr-bls-ensemble",
    )
    tts = RivaTTSService(
        server="localhost:50051",
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id="English-US.Female-1",
        language="en-US",
        quality=20,
        sample_rate=16000,
        model="fastpitch-hifigan-tts",
    )
    # Used to synchronize the user and bot transcripts in the UI
    stt_transcript_synchronization = UserTranscriptSynchronization()
    tts_transcript_synchronization = BotTranscriptSynchronization()

    # Uncomment the following if you want to use ElevenLabsTTS (make sure to comment out Riva TTS below)
    # tts = ElevenLabsTTSService(
    #     api_key=os.getenv("ELEVENLABS_API_KEY"),
    #     voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
    #     sample_rate=16000,
    #     model = "eleven_flash_v2_5",
    # )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Always answer as helpful friendly and polite. "
            "Respond with one sentence or less than 75 characters. Do not respond with bulleted or numbered list. "
            "Your output will be converted to audio so don't include special characters in your answers.",
        },
    ]

    context = OpenAILLMContext(messages)

    # Comment out the below line when enabling Speculative Speech Processing
    context_aggregator = llm.create_context_aggregator(context)

    # Uncomment the below line to enable speculative speech processing
    # nvidia_context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
    # Uncomment the below line to enable speculative speech processing
    # nvidia_tts_response_cacher = NvidiaTTSResponseCacher()

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            stt_transcript_synchronization,
            # Comment out the below line when enabling Speculative Speech Processing
            context_aggregator.user(),
            # Uncomment the below line to enable speculative speech processing
            # nvidia_context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            # Caches TTS responses for coordinated delivery in speculative
            # speech processing
            # nvidia_tts_response_cacher, # Uncomment to enable speculative speech processing
            tts_transcript_synchronization,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
            # Uncomment the below line to enable speculative speech processing
            # nvidia_context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            start_metadata={"stream_id": pipeline_metadata.stream_id},
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    return task


app = FastAPI()
app.include_router(websocket_router)
runner = ACEPipelineRunner(pipeline_callback=create_pipeline_task)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../static")), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
