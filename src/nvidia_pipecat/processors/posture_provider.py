# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Manages the ACE avatar's posture states during conversations."""

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame, StartInterruptionFrame, TTSStartedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.action import StartPostureBotActionFrame


class PostureProviderProcessor(FrameProcessor):
    """Controls the ACE avatar's posture based on conversation state transitions.

    This processor automatically manages the avatar's posture by transitioning between
    three states: Listening, Talking, and Attentive. The transitions are triggered by
    specific conversation events represented by input frames.

    Input Frames:
        TTSStartedFrame: Triggers transition to "Talking" posture
        StartInterruptionFrame: Triggers transition to "Listening" posture
        BotStoppedSpeakingFrame: Triggers transition to "Attentive" posture

    Output Frames:
        StartPostureBotActionFrame: Contains the target posture state
    """

    def __init__(self):
        """Initialize the PostureProviderProcessor.

        The processor starts with no initial state and will transition based on
        the first received frame.
        """
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame and update the avatar's posture if needed.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            await self.push_frame(StartPostureBotActionFrame("Talking"))
        if isinstance(frame, StartInterruptionFrame):
            await self.push_frame(StartPostureBotActionFrame("Listening"))
        if isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(StartPostureBotActionFrame("Attentive"))
