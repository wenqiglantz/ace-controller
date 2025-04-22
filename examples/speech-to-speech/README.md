# Speech to Speech Example

In this example, we showcase how to build a simple speech-to-speech voice assistant pipeline using nvidia-pipecat along with pipecat-ai library and deploy for testing. This pipeline uses a Websocket based ACETransport, Riva ASR and TTS models and NVIDIA LLM Service. We recommend first following [the Pipecat documentation](https://docs.pipecat.ai/getting-started/core-concepts) or [the ACE Controller](https://docs.nvidia.com/ace/ace-controller-microservice/latest/user-guide.html#pipecat-overview) Pipecat overview section to understand core concepts.

## Setup the environment

From the example directory, run the following commands to create a virtual environment and install the dependencies:

```bash
uv venv
uv sync
source .venv/bin/activate
```

Update the secrets in the `.env` file. Update all required environment variables for the necessary API keys. By default, we need NVIDIA_API_KEY to access NVIDIA LLM Services and you can get API key from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct).

```bash
cp env.example .env # and add your credentials
```

## Deploy local Riva ASR and TTS models.

#### Prerequisites
- You have access and are logged into NVIDIA NGC. For step-by-step instructions, refer to [the NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).

- You have access to an NVIDIA Volta™, NVIDIA Turing™, or an NVIDIA Ampere architecture-based A100 GPU. For more information, refer to [the Support Matrix](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix.html#support-matrix).

- You have Docker installed with support for NVIDIA GPUs. For more information, refer to [the Support Matrix]((https://docs.nvidia.com/deeplearning/riva/user-guide/docs/support-matrix.html#support-matrix)).

#### Download Riva Quick Start

Go to the Riva Quick Start for [Data center](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart/files?version=2.19.0). Select the File Browser tab to download the scripts or use [the NGC CLI tool](https://ngc.nvidia.com/setup/installers/cli) to download from the command line.

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart:2.19.0
```

#### Deploy Riva Speech Server

From the example directory, run below commands:

```bash
cd riva_quickstart_v2.19.0
chmod +x riva_init.sh riva_clean.sh riva_start.sh
bash riva_clean.sh ../utils/riva_config.sh
bash riva_init.sh ../utils/riva_config.sh
bash riva_start.sh ../utils/riva_config.sh
cd ..
```

This may take few minutes for the first time and will start the riva server on `localhost:50051`. This will deploy Riva Parakeet ASR and Riva Fastpitch TTS model.

For more info, you can refer to the [Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html).

## Using Elevenlabs TTS

1. Generate and set the `ELEVENLABS_API_KEY` in the `.env` file
2. Follow instructions in [bot.py](bot.py) and uncomment `ElevenLabsTTSService` import and usage.
3. Comment out the Riva TTS service



## Using Local NvidiaLLMService

By default, it connects to a hosted NIM, but can be configured to connect to a local NIM by setting the `base_url` parameter in `NvidiaLLMService` to the locally deployed LLM endpoint ( For example: base_url = http://machine_ip:port/v1 ). An API key is required to connect to the hosted NIM. For local LLM deployment, follow instructions from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct/deploy).

## Running the bot pipeline

After making all required changes, you can deploy the pipeline using below command

```bash
python bot.py
```

This will host the static web client along with the ACE controller server, visit `http://WORKSTATION_IP:8100/static/index.html` in your browser to start a session.

Note: For mic access, you will need to update chrome://flags/ and add http://WORKSTATION_IP:8100 in Insecure origins treated as secure section.

If you want to update the port, make changes in the `uvicorn.run` command in [the bot.py](bot.py) and the `wsUrl` in [the static/index.html](../static/index.html).

