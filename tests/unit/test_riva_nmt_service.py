# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the Riva Neural Machine Translation (NMT) service.

This module contains tests that verify the behavior of the RivaNMTService,
including successful translations and various error cases for both STT and LLM outputs.
"""

import pytest
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.transcriptions.language import Language

from nvidia_pipecat.services.riva_nmt import RivaNMTService
from tests.unit.utils import FrameStorage, ignore_ids, run_interactive_test


class MockText:
    """Mock class representing a translated text response.

    Attributes:
        text: The translated text content.
    """

    def __init__(self, text):
        """Initialize MockText.

        Args:
            text (str): The translated text content.
        """
        self.text = text


class MockTranslations:
    """Mock class representing a collection of translations.

    Attributes:
        translations: List of MockText objects containing translations.
    """

    def __init__(self, text):
        """Initialize MockTranslations.

        Args:
            text (str): The text to be wrapped in a MockText object.
        """
        self.translations = [MockText(text)]


class MockRivaNMTClient:
    """Mock class simulating the Riva NMT client.

    Attributes:
        translated_text: The text to return as translation.
    """

    def __init__(self, translated_text):
        """Initialize MockRivaNMTClient.

        Args:
            translated_text (str): Text to be returned as translation.
        """
        self.translated_text = translated_text

    def translate(self, arg1, arg2, arg3, arg4):
        """Mock translation method.

        Args:
            arg1: Source language.
            arg2: Target language.
            arg3: Text to translate.
            arg4: Additional options.

        Returns:
            MockTranslations: A mock translations object containing the pre-defined translated text.
        """
        return MockTranslations(self.translated_text)


@pytest.mark.asyncio()
async def test_riva_nmt_service(mocker):
    """Test the RivaNMTService functionality.

    Tests translation service behavior including successful translations
    and error handling for both STT and LLM outputs.

    Args:
        mocker: Pytest mocker fixture for mocking dependencies.

    The test verifies:
        - STT output translation
        - LLM output translation
        - Empty input handling
        - Missing language handling
        - Error frame generation
        - Frame sequence correctness
    """
    testcases = {
        "Success: STT output translated": {
            "source_language": Language.ES_US,
            "target_language": Language.EN_US,
            "input_frames": [TranscriptionFrame("Hola, por favor preséntate.", "", "")],
            "translated_text": "Hello, please introduce yourself.",
            "result_frame_name": "LLMMessagesFrame",
            "result_frame": LLMMessagesFrame([{"role": "system", "content": "Hello, please introduce yourself."}]),
        },
        "Success: LLM output translated": {
            "source_language": Language.EN_US,
            "target_language": Language.ES_US,
            "input_frames": [
                LLMFullResponseStartFrame(),
                TextFrame("Hello there!"),
                TextFrame("Im an artificial intelligence model known as Llama."),
                LLMFullResponseEndFrame(),
            ],
            "translated_text": "Hola Im un modelo de inteligencia artificial conocido como Llama",
            "result_frame_name": "TextFrame",
            "result_frame": TextFrame("Hola Im un modelo de inteligencia artificial conocido como Llama."),
        },
        "Fail due to empty input text": {
            "source_language": Language.ES_US,
            "target_language": Language.EN_US,
            "input_frames": [TranscriptionFrame("", "", "")],
            "translated_text": None,
            "result_frame_name": "ErrorFrame",
            "result_frame": ErrorFrame(
                f"Error while translating the text from {Language.ES_US} to {Language.EN_US}, "
                "Error: No input text provided for the translation..",
            ),
        },
        "Fail due to no source language provided": {
            "source_language": None,
            "target_language": Language.EN_US,
            "input_frames": [TranscriptionFrame("Hola, por favor preséntate.", "", "")],
            "translated_text": None,
            "error": Exception("No source language provided for the translation.."),
        },
        "Fail due to no target language provided": {
            "source_language": Language.ES_US,
            "target_language": None,
            "input_frames": [TranscriptionFrame("Hola, por favor preséntate.", "", "")],
            "translated_text": None,
            "error": Exception("No target language provided for the translation.."),
        },
    }

    for tc_name, tc_data in testcases.items():
        logger.info(f"Verifying test case: {tc_name}")

        mocker.patch(
            "riva.client.NeuralMachineTranslationClient", return_value=MockRivaNMTClient(tc_data["translated_text"])
        )

        try:
            nmt_service = RivaNMTService(
                source_language=tc_data["source_language"], target_language=tc_data["target_language"]
            )
        except Exception as e:
            assert str(e) == str(tc_data["error"])
            continue

        storage1 = FrameStorage()
        storage2 = FrameStorage()

        pipeline = Pipeline([storage1, nmt_service, storage2])

        async def test_routine(task: PipelineTask, test_data=tc_data, s1=storage1, s2=storage2):
            await task.queue_frames(test_data["input_frames"])

            # Wait for the result frame
            if "ErrorFrame" in test_data["result_frame"].name:
                await s1.wait_for_frame(ignore_ids(test_data["result_frame"]))
            else:
                await s2.wait_for_frame(ignore_ids(test_data["result_frame"]))

        await run_interactive_test(pipeline, test_coroutine=test_routine)

        for frame_history_entry in storage1.history:
            if frame_history_entry.frame.name.startswith("TextFrame"):
                # ignoring input text frames getting stored in storage1
                continue
            if frame_history_entry.frame.name.startswith(tc_data["result_frame_name"]):
                assert frame_history_entry.frame == ignore_ids(tc_data["result_frame"])

        for frame_history_entry in storage2.history:
            if frame_history_entry.frame.name.startswith(tc_data["result_frame_name"]):
                assert frame_history_entry.frame == ignore_ids(tc_data["result_frame"])
