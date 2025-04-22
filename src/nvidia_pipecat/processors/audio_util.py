# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Audio utilities."""

import os
import wave
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import AudioRawFrame, Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams

# ruff: noqa: SIM115


class AudioRecorder(FrameProcessor):
    """Records audio frames to a file.

    Args:
        output_file (str): The path to the output file.
        params (TransportParams): The transport parameters.
        bit_per_sample (int): The number of bits per sample. Defaults to 2.
        **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
    """

    def __init__(self, output_file: str, params: TransportParams, bit_per_sample: int = 2, **kwargs):
        """Initialize the AudioRecorder.

        Args:
            output_file (str): The path to the output file.
            params (TransportParams): The transport parameters.
            bit_per_sample (int): The number of bits per sample.
            **kwargs: Additional keyword arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._output_file = output_file
        Path(os.path.dirname(self._output_file)).mkdir(parents=True, exist_ok=True)
        self._writer = wave.open(self._output_file, "wb")
        self._writer.setnchannels(params.audio_out_channels)
        self._writer.setsampwidth(bit_per_sample)  # 2 bytes - 16 bits PCM
        self._writer.setframerate(params.audio_out_sample_rate)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        logger.debug(f"AudioFileSaver::process_frame - {frame}")
        if isinstance(frame, AudioRawFrame | TTSAudioRawFrame):
            logger.debug(f"writing audio frame (length: {len(frame.audio)})")
            self._writer.writeframes(frame.audio)

        await super().push_frame(frame, direction)

    async def cleanup(self):
        """Clean up the audio recorder.

        Closes the audio file writer and performs necessary cleanup operations.
        """
        await super().cleanup()
        logger.info("Finalizing audio file.")
        if self._writer:
            self._writer.close()
            self._writer = None
