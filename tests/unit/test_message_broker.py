# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""Unit tests for the MessageBroker."""

import asyncio
from datetime import timedelta

import pytest

from nvidia_pipecat.utils.message_broker import LocalQueueMessageBroker


async def send_task(mb: LocalQueueMessageBroker) -> bool:
    """Send test messages to the message broker.

    Args:
        mb: The LocalQueueMessageBroker instance to send messages to.

    Returns:
        bool: True if all messages were sent successfully.
    """
    for i in range(5):
        if i % 2 == 0:
            await mb.send_message("first", f"message {i}")
        else:
            await mb.send_message("second", f"message {i}")
        await asyncio.sleep(0.08)
    return True


async def receive_task(mb: LocalQueueMessageBroker) -> list[str]:
    """Receive messages from the message broker.

    Args:
        mb: The LocalQueueMessageBroker instance to receive messages from.

    Returns:
        list[str]: List of received message data.
    """
    result: list[str] = []
    while len(result) < 5:
        messages = await mb.receive_messages(timeout=timedelta(seconds=0.1))
        for _, message_data in messages:
            result.append(message_data)
    return result


@pytest.mark.asyncio()
async def test_local_queue_message_broker():
    """Tests basic concurrent send/receive functionality.

    Tests that messages can be successfully sent and received when send and
    receive tasks run concurrently.

    The test verifies:
        - All messages are received in order
        - Send operation completes successfully
        - Messages are correctly routed between channels
    """
    mb = LocalQueueMessageBroker(channels=["first", "second"])
    task_1 = asyncio.create_task(send_task(mb))
    task_2 = asyncio.create_task(receive_task(mb))

    results = await asyncio.gather(task_1, task_2)

    assert results == [True, ["message 0", "message 1", "message 2", "message 3", "message 4"]]


@pytest.mark.asyncio()
async def test_local_queue_message_broker_receive_first():
    """Tests message delivery when receive starts before send.

    Tests that no messages are lost when the receive task is started before
    any messages are sent.

    The test verifies:
        - All messages are received in order despite delayed send
        - Send operation completes successfully
        - No messages are lost due to timing
    """
    mb = LocalQueueMessageBroker(channels=["first", "second"])
    task_2 = asyncio.create_task(receive_task(mb))
    await asyncio.sleep(0.2)
    task_1 = asyncio.create_task(send_task(mb))

    results = await asyncio.gather(task_1, task_2)

    assert results == [True, ["message 0", "message 1", "message 2", "message 3", "message 4"]]


@pytest.mark.asyncio()
async def test_local_queue_message_broker_send_first():
    """Tests message delivery when send completes before receive starts.

    Tests that no messages are lost when all messages are sent before the
    receive task begins.

    The test verifies:
        - All messages are received in order despite delayed receive
        - Send operation completes successfully
        - Messages are properly queued until received
    """
    mb = LocalQueueMessageBroker(channels=["first", "second"])
    task_1 = asyncio.create_task(send_task(mb))
    await asyncio.sleep(0.5)
    task_2 = asyncio.create_task(receive_task(mb))

    results = await asyncio.gather(task_1, task_2)

    assert results == [True, ["message 0", "message 1", "message 2", "message 3", "message 4"]]
