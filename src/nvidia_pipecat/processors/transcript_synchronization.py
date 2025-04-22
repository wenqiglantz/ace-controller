# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Transcript synchronization processor."""

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.frames.transcripts import (
    BotUpdatedSpeakingTranscriptFrame,
    UserStoppedSpeakingTranscriptFrame,
    UserUpdatedSpeakingTranscriptFrame,
)


class UserTranscriptSynchronization(FrameProcessor):
    """Synchronizes user speech transcription events across the pipeline.

    This class synchronizes and exposes user speech transcripts by generating UserUpdatedSpeakingFrame
    and UserStoppedSpeakingTranscriptFrame events. It filters high frequency ASR frames to only forward
    transcripts that contain actual changes. The final transcript is sent with a
    UserStoppedSpeakingTranscriptFrame.

    Attributes:
        partial_transcript (str | None): Current partial ASR transcript.
        final_transcript (str | None): Final ASR transcript.
        stopped_speaking (bool | None): Whether user has stopped speaking.

    Input Frames:
        InterimTranscriptionFrame: ASR partial transcript
        TranscriptionFrame: ASR final transcript
        UserStartedSpeakingFrame: User starts speaking
        UserStoppedSpeakingFrame: User stops speaking
        StartInterruptionFrame: Resets processor state
    """

    def __init__(self):
        """Initializes the processor with default state values."""
        super().__init__()
        self.partial_transcript: str | None = None
        self.final_transcript: str | None = None
        self.stopped_speaking: bool | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes frames and updates transcript state.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            # Check if partial transcript changed
            # TODO: We need filter out more of the duplicated or very similar transcripts with Riva TTS
            if not self.partial_transcript or self.partial_transcript != frame.text:
                self.partial_transcript = frame.text
                await self.push_frame(UserUpdatedSpeakingTranscriptFrame(transcript=frame.text), direction)
        elif isinstance(frame, TranscriptionFrame):
            self.final_transcript = frame.text
        elif isinstance(frame, UserStartedSpeakingFrame):
            self.stopped_speaking = False
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.stopped_speaking = True
        elif isinstance(frame, StartInterruptionFrame):
            self.partial_transcript = None
            self.final_transcript = None
            self.stopped_speaking = None

        # We wait until we received both, TranscriptionFrame and UserStoppedSpeakingFrame
        if self.final_transcript and self.stopped_speaking:
            await self.push_frame(UserStoppedSpeakingTranscriptFrame(self.final_transcript), direction)
            self.partial_transcript = None
            self.final_transcript = None
            self.stopped_speaking = None


class BotTranscriptSynchronization(FrameProcessor):
    """Synchronizes bot speech transcripts with audio playback.

    Synchronizes TTSTextFrames with BotStartedSpeakingFrame events that indicate the start
    of speech audio playback. Creates BotUpdatedSpeakingTranscriptFrame frames containing
    partial and final transcripts.

    The bot transcription synchronization is based on the assumption that there is a
    pair of BotStartedSpeakingFrame and BotStoppedSpeakingFrame for each TTSStartedFrame.
    This processor will not work correctly if these assumptions are not met.
    Each BotStartedSpeakingFrame will trigger a BotUpdatedSpeakingTranscriptFrame if a
    transcript is available. It is ok that multiple [TTSStartedFrame, TTSTextFrame, ...]
    sequences come before the corresponding BotStoppedSpeakingFrame.

    Input Frames:
        TTSStartedFrame: Speech audio playback starts
        TTSTextFrame: Text to be spoken
        BotStartedSpeakingFrame: Bot starts speaking
        BotStoppedSpeakingFrame: Bot stops speaking
        StartInterruptionFrame: Resets processor state
    """

    def __init__(self):
        """Initialize the BotTranscriptSynchronization processor."""
        super().__init__()
        self.bot_started_speaking = False
        self.bot_transcripts_buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes frames and manages transcript synchronization.

        Args:
            frame (Frame): Incoming frame to process.
            direction (FrameDirection): Frame flow direction.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            # Reset transcript buffer
            self.bot_transcripts_buffer = []
            self.bot_started_speaking = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            # Start buffering the next transcript
            self.bot_transcripts_buffer.append("")
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSTextFrame):
            # Aggregate partial transcripts
            if not self.bot_transcripts_buffer:
                logger.warning("TTSTextFrame received before TTSStartedFrame!")
                # Likely due to a2f crashes. This is a fallback to keep things working.
                self.bot_transcripts_buffer.append("")
            if frame.text:
                if self.bot_transcripts_buffer[-1]:
                    self.bot_transcripts_buffer[-1] += f" {frame.text}"
                else:
                    self.bot_transcripts_buffer[-1] = frame.text
            # TODO: We need to figure out how to align the partial transcripts
            # with the audio. Currently, they are shown as soon as they come in.
            # This is needed to get the full transcript from Elevenlabs TTS
            # since it creates TTSStartedFrame right with the first incoming word
            # and does not stop until all the text is generated!
            if self.bot_started_speaking and len(self.bot_transcripts_buffer) == 1:
                # Only create BotUpdatedSpeakingTranscriptFrame if
                # BotStartedSpeakingFrame is related to the first active transcript
                await self.push_frame(
                    BotUpdatedSpeakingTranscriptFrame(transcript=self.bot_transcripts_buffer[-1].strip()), direction
                )
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Start synchronizing transcript from buffer with BotStartedSpeakingFrame
            if self.bot_started_speaking:
                logger.warning("BotStartedSpeakingFrame received before BotStoppedSpeakingFrame!")
            self.bot_started_speaking = True
            await self.push_frame(frame, direction)
            if self.bot_transcripts_buffer and self.bot_transcripts_buffer[0]:
                # If already available push the transcript
                await self.push_frame(
                    BotUpdatedSpeakingTranscriptFrame(self.bot_transcripts_buffer[0]), FrameDirection.DOWNSTREAM
                )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Remove the shown transcript from the buffer
            if not self.bot_started_speaking:
                logger.warning("BotStoppedSpeakingFrame received before BotStartedSpeakingFrame!")
            self.bot_started_speaking = False
            if self.bot_transcripts_buffer:
                self.bot_transcripts_buffer.pop(0)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
