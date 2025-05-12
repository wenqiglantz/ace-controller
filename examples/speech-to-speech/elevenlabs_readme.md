# ElevenLabs Speech-to-Speech Example

This example demonstrates how to build a speech-to-speech voice assistant pipeline using nvidia-pipecat with ElevenLabs speech services. The pipeline uses:

- ACE Transport for audio I/O
- ElevenLabsASRService for speech recognition
- NvidiaLLMService for text generation
- ElevenLabsTTSServiceWithEndOfSpeech for speech synthesis

## Features

- Uses ElevenLabs Scribe v1 for high-quality speech recognition
- Supports multiple languages and speaker diarization
- Provides word-level timestamps
- Uses ElevenLabs voice synthesis for natural-sounding responses
- Integrates with NVIDIA LLM Service for text generation

## Prerequisites

To run this example, you need:

1. An ElevenLabs API key (sign up at [elevenlabs.io](https://elevenlabs.io))
2. An NVIDIA API key for access to NVIDIA LLM Service (get from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct))
3. Python 3.10+ with pip

## Setup

1. From the example directory, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
ELEVENLABS_VOICE_ID=optional_custom_voice_id  # Optional, defaults to "Rachel"
```

## Running the Example

Start the server with:

```bash
python elevenlabs_example.py
```

Then visit `http://localhost:8100/static/index.html` in your browser to start a conversation.

## Configuration Options

The example includes several configuration options:

### ElevenLabs ASR Service

```python
stt = ElevenLabsASRService(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    model="scribe_v1",                # Can also use "scribe_v1_experimental"
    language="en-US",                 # Optional, auto-detected if not specified
    sample_rate=16000,                # Audio sample rate in Hz
    diarize=False,                    # Enable speaker diarization
    tag_audio_events=True,            # Tag audio events like laughter
    timestamps_granularity="word",    # "word" or "character" level timestamps
    chunk_size_seconds=3,             # Buffer size before sending to API
)
```

### ElevenLabs TTS Service

```python
tts = ElevenLabsTTSServiceWithEndOfSpeech(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="EXAVITQu4vr4xnSDxMaL",  # Rachel voice, customize as needed
    sample_rate=16000,                # Audio sample rate in Hz
    model="eleven_turbo_v2",          # Can also use "eleven_multilingual_v2" or "eleven_flash_v2"
)
```

## Notes on Implementation

- The ElevenLabs ASR service uses a buffering approach, collecting audio in chunks before sending to the API, as the API is designed for processing complete audio files.
- For a more interactive experience, the chunk size can be reduced, though this may result in more API calls.
- The service supports multiple languages automatically detected by the Scribe v1 model.

## Limitations

- ElevenLabs ASR is not designed for real-time streaming like Riva ASR, so there may be slight delays in transcription.
- API quota limitations apply based on your ElevenLabs subscription tier.

## Troubleshooting

- If you encounter issues with audio input, make sure your browser allows microphone access.
- For Chrome, you may need to add your localhost URL to "Insecure origins treated as secure" in chrome://flags/.
- Check the server logs for detailed error messages from the ElevenLabs or NVIDIA APIs. 