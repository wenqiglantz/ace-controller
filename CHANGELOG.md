# NVIDIA Pipecat 0.1.0 (23 April 2025)
The NVIDIA Pipecat library augments the Pipecat framework by adding additional frame processors and services, as well as new multimodal frames to enhance avatar interactions. This is the first release of the NVIDIA Pipecat library.

## New Features

- Added Pipecat services for [Riva ASR (Automatic Speech Recognition)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html#), [Riva TTS (Text to Speech)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html), and [Riva NMT (Neural Machine Translation)](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/translation/translation-overview.html) models.
- Added Pipecat frames, processors, and services to support multimodal avatar interactions and use cases. This includes `Audio2Face3DService`, `AnimationGraphService`, `FacialGestureProviderProcessor`, and `PostureProviderProcessor`.
- Added `ACETransport`, which is specifically designed to support integration with existing [ACE microservices](https://docs.nvidia.com/ace/overview/latest/index.html). This includes a FastAPI-based HTTP and WebSocket server implementation compatible with ACE.
- Added `NvidiaLLMService` for [NIM LLM models](https://build.nvidia.com/) and `NvidiaRAGService` for the [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/quickstart.md).
- Added `UserTranscriptSynchronization` processor for user speech transcripts and `BotTranscriptSynchronization` processor for synchronizing bot transcripts with bot audio playback.
- Added custom context aggregators and processors to enable [Speculative Speech Processing](https://docs.nvidia.com/ace/ace-controller-microservice/latest/user-guide.html#speculative-speech-processing) to reduce latency.
- Added `UserPresence`, `Proactivity`, and `AcknowledgementProcessor` frame processors to improve human-bot interactions.
- Released source code for the voice assistant example using `nvidia-pipecat`, along with the `pipecat-ai` library service, to showcase NVIDIA services with `ACETransport`.


## Improvements

- Added `ElevenLabsTTSServiceWithEndOfSpeech`, an extended version of the ElevenLabs TTS service with end-of-speech events for usage in avatar interactions.

