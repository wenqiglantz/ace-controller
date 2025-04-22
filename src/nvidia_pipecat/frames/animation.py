# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Animation frames."""

from dataclasses import dataclass

from nvidia_ace.animation_pb2 import AnimationData, SkelAnimationHeader
from nvidia_ace.audio_pb2 import AudioHeader
from pipecat.frames.frames import ControlFrame, DataFrame

from nvidia_pipecat.frames.action import ActionFrame


@dataclass
class AnimationDataStreamStartedFrame(ControlFrame, ActionFrame):
    """An animation data stream has started. Contains both animation and audio headers.

    Args:
        animation_source_id: Where is the animation generated from. Useful if using an animation graph.
        audio_header: Audio header for the stream.
        animation_header: Animation header for the stream.
    """

    animation_source_id: str
    audio_header: AudioHeader
    animation_header: SkelAnimationHeader


@dataclass
class AnimationDataStreamRawFrame(DataFrame, ActionFrame):
    """Animation data that may contain both audio and/or animation data.

    Args:
        animation_data: Animation data containing audio and/or animation content. If both are
            present, the animations in the frame are synced with the audio.
    """

    animation_data: AnimationData


@dataclass
class AnimationDataStreamStoppedFrame(ControlFrame, ActionFrame):
    """Signals the end of the animation data stream."""
