# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Tracing utilities."""

import asyncio
import contextlib
import enum
import functools
import inspect
from collections.abc import Callable

import opentelemetry.trace
from opentelemetry import metrics, trace


class Traceable:
    """Base class for traceable objects.

    This class provides tracing functionality using OpenTelemetry. It initializes tracing and metrics
    components and manages spans for tracing operations.
    """

    def __init__(self, name: str, **kwargs):
        """Initialize a traceable object.

        Args:
            name (str): Name of the traceable object used for the initial span
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._tracer = trace.get_tracer("traceable")
        self._meter = metrics.get_meter("meter")
        self._parent_span_id = trace.get_current_span().get_span_context().span_id
        self._span = self._tracer.start_span(name)
        self._span.end()

    @property
    def meter(self):
        """Returns the OpenTelemetry meter instance.

        The meter is used for collecting metrics and measurements in OpenTelemetry.

        Returns:
            opentelemetry.metrics.Meter: The OpenTelemetry meter instance used by this traceable object.
        """
        return self._meter


class AttachmentStrategy(enum.Enum):
    """Attachment strategy for the @traced annotation.

    CHILD: span will be attached to the class span if no parent or to its parent otherwise.
    LINK : span will be attached to the class span but linked to its parent.
    NONE : span will be attached to the class span even if nested under another span.
    """

    CHILD = enum.auto()
    LINK = enum.auto()
    NONE = enum.auto()


@contextlib.contextmanager
def __traced_context_manager(
    self: Traceable, func: Callable, name: str | None, attachment_strategy: AttachmentStrategy
):
    if not isinstance(self, Traceable):
        raise RuntimeError("@traced annotation can only be used in classes inheriting from Traceable")
    stack = contextlib.ExitStack()
    try:
        current_span = trace.get_current_span()
        is_span_class_parent_span = current_span.get_span_context().span_id == self._parent_span_id
        match attachment_strategy:
            case AttachmentStrategy.CHILD if not is_span_class_parent_span:
                stack.enter_context(self._tracer.start_as_current_span(func.__name__ if name is None else name))  # type: ignore
            case AttachmentStrategy.LINK:
                if is_span_class_parent_span:
                    link = trace.Link(self._span.get_span_context())
                else:
                    link = trace.Link(current_span.get_span_context())
                stack.enter_context(opentelemetry.trace.use_span(span=self._span, end_on_exit=False))  # type: ignore
                stack.enter_context(
                    self._tracer.start_as_current_span(func.__name__ if name is None else name, links=[link])
                )  # type: ignore
            case AttachmentStrategy.NONE | AttachmentStrategy.CHILD:
                stack.enter_context(opentelemetry.trace.use_span(span=self._span, end_on_exit=False))  # type: ignore
                stack.enter_context(self._tracer.start_as_current_span(func.__name__ if name is None else name))  # type: ignore
        yield
    finally:
        stack.close()


def __traced_decorator(func, name, attachment_strategy: AttachmentStrategy):
    @functools.wraps(func)
    async def coroutine_wrapper(self: Traceable, *args, **kwargs):
        exception = None
        with __traced_context_manager(self, func, name, attachment_strategy):
            try:
                return await func(self, *args, **kwargs)
            except asyncio.CancelledError as e:
                exception = e
        if exception:
            raise exception

    @functools.wraps(func)
    async def generator_wrapper(self: Traceable, *args, **kwargs):
        exception = None
        with __traced_context_manager(self, func, name, attachment_strategy):
            try:
                async for v in func(self, *args, **kwargs):
                    yield v
            except asyncio.CancelledError as e:
                exception = e
        if exception:
            raise exception

    if inspect.iscoroutinefunction(func):
        return coroutine_wrapper
    if inspect.isasyncgenfunction(func):
        return generator_wrapper

    raise ValueError("@traced annotation can only be used on async or async generator functions")


def traced(
    func: Callable | None = None,
    *,
    name: str | None = None,
    attachment_strategy: AttachmentStrategy = AttachmentStrategy.CHILD,
):
    """Decorator that adds tracing to an async function.

    This decorator wraps an async function to add OpenTelemetry tracing capabilities.
    It creates a new span for the function and maintains proper context propagation.
    For FrameProcessor process_frame method, it also ensures the parent span is properly set.

    Args:
        func: The async function to be traced.
        name: Name for the span. Defaults to the name of the function.
        attachment_strategy (AttachmentStrategy): The attachment strategy to use (Possible values are
        CHILD, LINK, NONE)

    Returns:
        A wrapped async function with tracing capabilities.

    Raises:
        RuntimeError: If used on a function in a class that doesn't inherit from Traceable.

    Example:
        @traceable
        class MyClass:
            @traced
            async def my_function(self):
                pass
    """
    if func is not None:
        return __traced_decorator(func, name=name, attachment_strategy=attachment_strategy)
    else:
        return functools.partial(__traced_decorator, name=name, attachment_strategy=attachment_strategy)


def traceable(cls):
    """Class decorator that makes a class traceable for OpenTelemetry instrumentation.

    This decorator creates a new class that inherits from both the original class and
    the Traceable base class. The new class will be ready to be instrumented for tracing using
    the @traced decorator.

    Args:
        cls: The class to make traceable.

    Returns:
        TracedClass: A new class that inherits from both the input class and Traceable,
            with tracing capabilities added.

    Example:
        @traceable
        class MyClass:
            @traced
            def my_method(self):
                pass
    """

    @functools.wraps(cls, updated=())
    class TracedClass(cls, Traceable):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            if hasattr(self, "name"):
                Traceable.__init__(self, self.name)
            else:
                Traceable.__init__(self, cls.__name__)

    return TracedClass
