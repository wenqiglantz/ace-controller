# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests tracing."""

import pytest
from opentelemetry import trace
from pipecat.frames.frames import EndFrame, ErrorFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nvidia_pipecat.utils.tracing import AttachmentStrategy, traceable, traced
from tests.unit.utils import ignore_ids, run_test


@pytest.mark.asyncio
async def test_traced_processor_basic_usage():
    """Tests basic tracing functionality in a pipeline processor.

    Tests the @traceable and @traced decorators with different attachment
    strategies in a simple pipeline configuration.

    The test verifies:
        - Span creation and attachment
        - Event recording
        - Nested span handling
        - Generator tracing
        - Frame processing
    """
    tracer = trace.get_tracer(__name__)

    @traceable
    class TestProcessor(FrameProcessor):
        """Example processor demonstrating how to use the tracing utilities."""

        @traced(attachment_strategy=AttachmentStrategy.NONE)
        async def process_frame(self, frame, direction):
            """Process a frame with tracing.

            Args:
                frame: The frame to process.
                direction: The direction of frame flow.

            The method demonstrates:
                - Basic span creation
                - Event recording
                - Nested span handling
                - Multiple tracing strategies
            """
            await super().process_frame(frame, direction)
            trace.get_current_span().add_event("Before inner")
            with tracer.start_as_current_span("inner") as span:
                span.add_event("inner event")
                await self.child()
                await self.linked()
                await self.none()
            trace.get_current_span().add_event("After inner")
            async for f in self.generator():
                print(f"{f}")
            await super().push_frame(frame, direction)

        @traced
        async def child(self):
            """Example method with child attachment strategy.

            This span is attached as CHILD, meaning it will be attached to
            the class span if no parent exists, or to its parent otherwise.
            """
            trace.get_current_span().add_event("child")

        @traced(attachment_strategy=AttachmentStrategy.LINK)
        async def linked(self):
            """Example method with link attachment strategy.

            This span is attached as LINK, meaning it will be attached to
            the class span but linked to its parent.
            """
            trace.get_current_span().add_event("linked")

        @traced(attachment_strategy=AttachmentStrategy.NONE)
        async def none(self):
            """Example method with no attachment strategy.

            This span is attached as NONE, meaning it will be attached to
            the class span even if nested under another span.
            """
            trace.get_current_span().add_event("none")

        @traced
        async def generator(self):
            """Example generator method with tracing.

            Demonstrates tracing in a generator context.

            Yields:
                TextFrame: Text frames with sample content.
            """
            yield TextFrame("Hello, ")
            trace.get_current_span().add_event("generated!")
            yield TextFrame("World")

    processor = TestProcessor()
    pipeline = Pipeline([processor])

    await run_test(
        pipeline,
        frames_to_send=[],
        expected_down_frames=[],
    )


@pytest.mark.asyncio
async def test_wrong_usage() -> None:
    """Test that error message is raised when a processor is not traceable."""

    class TestProcessor(FrameProcessor):
        @traced
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            await super().push_frame(frame, direction)

    class HasSeenError(FrameProcessor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seen_error = False

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, ErrorFrame):
                self.seen_error = True
            elif isinstance(frame, EndFrame):
                assert self.seen_error
            await super().push_frame(frame, direction)

    seen_error = HasSeenError()
    processor = TestProcessor()
    pipeline = Pipeline([seen_error, processor])
    await run_test(
        pipeline,
        frames_to_send=[],
        expected_down_frames=[],
        expected_up_frames=[
            ignore_ids(ErrorFrame("@traced annotation can only be used in classes inheriting from Traceable"))
        ],
    )


@pytest.mark.asyncio
async def test_no_processor() -> None:
    """Test that a processor can be used without a pipeline."""

    @traceable
    class TestTraceable:
        @traced
        async def traced_test(self):
            trace.get_current_span().add_event("I can use it as another utility function as well.")

    test = TestTraceable()
    await test.traced_test()
