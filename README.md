# ACE Controller SDK

The ACE Controller SDK allows you to build your own ACE Controller service to manage multimodal, real-time interactions with voice bots and avatars using NVIDIA ACE. With the SDK, you can create controllers that leverage the Python-based open-source [Pipecat framework](https://github.com/pipecat-ai/pipecat) for creating real-time, voice-enabled, and multimodal conversational AI agents. The SDK contains enhancements to the Pipecat framework, enabling developers to effortlessly customize, debug, and deploy complex pipelines while integrating robust NVIDIA Services into the Pipecat ecosystem.

## Main Features

- **Pipecat Extension:** A Pipecat extension to connect with ACE services and NVIDIA NIMs, facilitating the creation of human-avatar interactions. The NVIDIA Pipecat library augments [the Pipecat framework](https://github.com/pipecat-ai/pipecat) by adding additional frame processors and services, as well as new multimodal frames to enhance avatar interactions. This includes the integration of NVIDIA services and NIMs such as [NVIDIA Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html), [NVIDIA Audio2Face](https://build.nvidia.com/nvidia/audio2face-3d), and [NVIDIA Foundational RAG](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline).

- **HTTP and WebSocket Server Implementation:** The SDK provides a FastAPI-based HTTP and WebSocket server implementation compatible with ACE. It includes functionality for stream and pipeline management by offering new Pipecat pipeline runners and transports. For ease of use and distribution, this functionality is currently included in the `nvidia-pipecat` Python library as well.

## ACE Controller Microservice

The ACE Controller SDK was used to build the [ACE Controller Microservice](https://docs.nvidia.com/ace/ace-controller-microservice/latest/index.html).Check out the [ACE documentation](https://docs.nvidia.com/ace/tokkio/latest/customization/customization-options.html) for more details on how to configure the ACE Controller MS with your custom pipelines.


## Getting Started

The NVIDIA Pipecat package is released as a wheel on PyPI. Create a Python virtual environment and use the pip command to install the nvidia-pipecat package.

```bash
pip install nvidia-pipecat
```

You can start building pipecat pipelines utilizing services from the NVIDIA Pipecat package. For more details, follow [the ACE Controller](https://docs.nvidia.com/ace/ace-controller-microservice/latest/index.html) and [the Pipecat Framework](https://docs.pipecat.ai/getting-started/overview) documentation.

## Hacking on the framework itself

If you wish to work directly with the source code or modify services from the nvidia-pipecat package, you can utilize either the UV or Nix development setup as outlined below.

### Using UV


To get started, first install the [UV package manager](https://docs.astral.sh/uv/#highlights). 

Then, create a virtual environment with all the required dependencies by running the following commands:
```bash
uv venv
uv sync
source .venv/bin/activate
```

Once the environment is set up, you can begin building pipelines or modifying the services in the source code.

If you wish to contribute your changes to the repository, please ensure you run the unit tests, linter, and formatting tool.

To run unit tests, use:
```
uv run pytest
```

To format the code, use:
```bash
ruff format
```

To run the linter, use:
```
ruff check
```


### Using Nix

To set up your development environment using [the Nix](https://nixos.org/download/#nix-install-linux), follow these steps:

Initialize the development environment: Simply run the following command:
```bash
nix develop
```

This setup provides you with a fully configured environment, allowing you to focus on development without worrying about dependency management.

To ensure that all checks such as the formatting and linter for the repository are passing, use the following command:

```bash
nix flake check
```

## CONTRIBUTING

We invite contributions! Open a GitHub issue or pull request! See contributing guildelines [here](./CONTRIBUTING.md).

