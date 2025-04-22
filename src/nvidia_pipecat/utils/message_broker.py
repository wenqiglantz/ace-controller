# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Message broker implementation.

This module provides message broker implementations for Redis and local queue-based
communication, enabling publish/subscribe patterns and key-value storage operations.
"""

import asyncio
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta

import redis.asyncio as redis
from loguru import logger
from redis.asyncio.client import PubSub


@dataclass
class MessageBrokerConfig:
    """Configuration for message broker initialization.

    Attributes:
        name: The type of message broker to use ('redis' or 'local_queue').
        url: Connection URL for the message broker (required for Redis).
    """

    name: str
    url: str = ""


class MessageBroker(ABC):
    """Abstract interface for all message brokers. Defines interface to receive and send messages."""

    @abstractmethod
    async def receive_messages(self, timeout: timedelta | None = timedelta(seconds=0.5)) -> list[tuple[str, str]]:
        """Receive incoming messages. Returns when it received one or more messages.

        Args:
            timeout: Maximum time to wait for messages. None means wait indefinitely.

        Returns:
            list[tuple[str, str]]: List of (message_id, message_data) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_message(self, channel_id: str, message_data: str) -> None:
        """Publish a message to a channel.

        Args:
            channel_id: The channel to publish to.
            message_data: The message content to publish.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Get value for a key.

        Args:
            key: The key to retrieve.

        Returns:
            str | None: The value if found, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def set(self, key: str, value: str) -> None:
        """Set value for a key.

        Args:
            key: The key to set.
            value: The value to store.
        """
        raise NotImplementedError

    @abstractmethod
    async def wait_for_connection(self) -> None:
        """Wait for the connection to the message broker to be established."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from storage.

        Args:
            key: The key to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_latest_message(self, channel_id: str) -> str | None:
        """Return the most recent message in the channel.

        Args:
            channel_id: The channel to check.

        Returns:
            str | None: The latest message if available, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def pubsub_receive_message(self, channels: list[str], timeout: timedelta | None = None) -> str | None:
        """Receive a message from specified channels using pub/sub pattern.

        Args:
            channels: List of channels to subscribe to.
            timeout: Maximum time to wait for a message.

        Returns:
            str | None: The received message if available, None otherwise.
        """
        raise NotImplementedError


class RedisMessageBroker(MessageBroker):
    """Message broker implementation using Redis.

    Provides interface to receive and send messages using Redis streams and pub/sub.
    """

    def __init__(self, redis_url: str, channels: list[str]):
        """Initialize Redis message broker.

        Args:
            redis_url: URL for Redis connection.
            channels: List of channels to subscribe to.
        """
        super().__init__()
        self.redis: redis.Redis = redis.from_url(redis_url)
        self._pubsub: PubSub | None = None
        self._channel_state: dict[str, str] = dict(map(lambda c: (c, "0"), channels))
        self.is_connected = asyncio.Event()

        # Add connection check
        asyncio.create_task(self._check_connection())

    async def wait_for_connection(self) -> None:
        """Wait for the connection to the message broker to be established."""
        await self.is_connected.wait()

    async def _check_connection(self) -> None:
        """Verify Redis connection is working."""
        try:
            await self.redis.ping()
            logger.info("Successfully connected to Redis")
            self.is_connected.set()
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected.clear()

    async def receive_messages(self, timeout: timedelta | None = timedelta(seconds=0.5)) -> list[tuple[str, str]]:
        """Receive incoming messages. Returns when it received one or more messages.

        Args:
            timeout: Maximum time to wait for messages.

        Returns:
            list[tuple[str, str]]: List of (message_id, message_data) tuples.
        """
        timeout_ms: int | None = None if timeout is None else int(timeout.total_seconds() * 1000)

        if timeout_ms is not None and timeout_ms < 100:
            logger.warning(f"Redis timeout resolution is about 100ms, but a timeout of {timeout_ms} ms was given.")

        result = await self.redis.xread(streams=self._channel_state, block=timeout_ms)  # type: ignore
        message_list: list[tuple[str, str]] = []

        for channel in result:
            channel_id = str(channel[0].decode())
            for message_id, value in channel[1]:
                message_id = message_id.decode()
                for key in value:
                    message_data = value[key].decode()
                    message_list.append((message_id, message_data))

                self._channel_state[channel_id] = message_id

        return message_list

    async def pubsub_receive_message(self, channels: list[str], timeout: timedelta | None = None) -> str | None:
        """Receive a message from specified channels using pub/sub pattern.

        Args:
            channels: List of channels to subscribe to.
            timeout: Maximum time to wait for a message.

        Returns:
            str | None: The received message if available, None otherwise.
        """
        if not self._pubsub:
            self._pubsub = self.redis.pubsub()

        for channel in channels:
            if channel.encode("utf-8") not in self._pubsub.channels:
                await self._pubsub.subscribe(channel)

        message = await self._pubsub.get_message(timeout=None)
        if message["type"] == "message":
            return message["data"].decode("utf-8")
        return None

    async def send_message(self, channel_id: str, message_data: str) -> None:
        """Publish a message to a channel.

        Args:
            channel_id: The channel to publish to.
            message_data: The message content to publish.
        """
        await self.redis.xadd(channel_id, {"event": message_data.encode()})

    async def get(self, key: str) -> str | None:
        """Get value for a key.

        Args:
            key: The key to retrieve.

        Returns:
            str | None: The value if found, None otherwise.
        """
        return await self.redis.get(key)

    async def set(self, key: str, value: str) -> None:
        """Set value for a key.

        Args:
            key: The key to set.
            value: The value to store.
        """
        await self.redis.set(name=key, value=value)

    async def delete(self, key: str) -> None:
        """Delete a key from storage.

        Args:
            key: The key to delete.
        """
        await self.redis.delete(key)

    async def get_latest_message(self, channel_id: str) -> str | None:
        """Return the most recent message in the channel.

        Args:
            channel_id: The channel to check.

        Returns:
            str | None: The latest message if available, None otherwise.
        """
        result = await self.redis.xrevrange(channel_id, count=1)

        if not result:
            return None

        _, message_data = result[0]

        try:
            for key in message_data:
                return message_data[key].decode()
            return None
        except Exception:
            # Ignore parsing errors, but log an info about it.
            logger.info(f"Latest message {message_data} on channel {channel_id} is no valid message")
            return None


_MESSAGE_ID = itertools.count()


class LocalQueueMessageBroker(MessageBroker):
    """Message broker using a local queue for testing purposes.

    Implements the MessageBroker interface using in-memory queues, suitable for testing
    and local development.
    """

    _storage: dict[str, str] = {}
    _queues: dict[str, asyncio.Queue] = {}

    def __init__(self, channels: list[str]):
        """Initialize local queue message broker.

        Args:
            channels: List of channels to create queues for.
        """
        super().__init__()
        self._update_channels(channels)

    def _update_channels(self, channels: list[str]) -> None:
        for channel in channels:
            if channel not in self._queues:
                self._queues[channel] = asyncio.Queue()

    async def receive_messages(
        self, timeout: timedelta | None = timedelta(seconds=0.5), channels: list[str] | None = None
    ) -> list[tuple[str, str]]:
        """Receive incoming messages. Returns when it received one or more messages.

        Args:
            timeout: Maximum time to wait for messages.
            channels: Optional list of specific channels to receive from.

        Returns:
            list[tuple[str, str]]: List of (message_id, message_data) tuples.
        """
        timeout_in_seconds = timeout.total_seconds() if timeout else None

        queue_selection = self._queues
        if channels is not None:
            self._update_channels(channels)
            queue_selection = {channel: queue for channel, queue in self._queues.items() if channel in channels}

        aws = [
            asyncio.create_task(asyncio.wait_for(q.get(), timeout=timeout_in_seconds)) for q in queue_selection.values()
        ]

        results = await asyncio.gather(*aws, return_exceptions=True)
        message_list: list[tuple[str, str]] = []
        for result in results:
            if isinstance(result, tuple):
                message_list.append(result)

        return message_list

    async def pubsub_receive_message(self, channels: list[str], timeout: timedelta | None = None) -> str | None:
        """Receive a message from specified channels using pub/sub pattern.

        Args:
            channels: List of channels to subscribe to.
            timeout: Maximum time to wait for a message.

        Returns:
            str | None: The received message if available, None otherwise.
        """
        messages = await self.receive_messages(timeout=timeout, channels=channels)
        return messages[0][1]

    async def send_message(self, channel_id: str, message_data: str) -> None:
        """Publish a message to a channel.

        Args:
            channel_id: The channel to publish to.
            message_data: The message content to publish.
        """
        message_id = str(next(_MESSAGE_ID))
        await self._queues.setdefault(channel_id, asyncio.Queue()).put((message_id, message_data))

    async def wait_for_connection(self) -> None:
        """Wait for the connection to the message broker to be established.

        For local queue, this is a no-op as no connection is needed.
        """

    async def get(self, key: str) -> str | None:
        """Get value for a key.

        Args:
            key: The key to retrieve.

        Returns:
            str | None: The value if found, None otherwise.
        """
        return self._storage.get(key, None)

    async def set(self, key: str, value: str) -> None:
        """Set value for a key.

        Args:
            key: The key to set.
            value: The value to store.
        """
        self._storage[key] = value

    async def delete(self, key: str) -> None:
        """Delete a key from storage.

        Args:
            key: The key to delete.
        """
        if key in self._queues:
            del self._queues[key]
        if key in self._storage:
            del self._storage[key]

    async def get_latest_message(self, channel_id: str) -> str | None:
        """Return the most recent message in the channel.

        Args:
            channel_id: The channel to check.

        Returns:
            str | None: The latest message if available, None otherwise.
        """
        return None


def message_broker_factory(config: MessageBrokerConfig, channels: list[str]) -> MessageBroker:
    """Create a new message broker based on the configuration.

    Currently supports Redis and local queue implementations.

    Args:
        config: Configuration specifying the broker type and connection details.
        channels: List of channels to initialize.

    Returns:
        MessageBroker: Configured message broker instance.

    Raises:
        Exception: If the specified broker type is not supported.
    """
    brokers = ["redis", "local_queue"]
    if config.name == "redis":
        return RedisMessageBroker(config.url, channels)
    elif config.name == "local_queue":
        return LocalQueueMessageBroker(channels)
    else:
        raise Exception(f"message broker {config.name} does not exist. Available providers {','.join(brokers)}")
