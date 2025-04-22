# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""HTTP client utilities for making REST API calls."""

import asyncio
import json
import logging
from enum import Enum
from http import HTTPStatus
from typing import Any, Final

import aiohttp

logger = logging.getLogger(__name__)


class CallMethod(str, Enum):
    """Enumeration of supported HTTP methods.

    Attributes:
        POST: HTTP POST method.
        PUT: HTTP PUT method.
        GET: HTTP GET method.
    """

    POST = "post"
    PUT = "put"
    GET = "get"


DEFAULT_TIMEOUT: Final[aiohttp.ClientTimeout] = aiohttp.ClientTimeout(total=10, sock_connect=5)


class HttpClient:
    """HTTP client for making REST API calls."""

    def __init__(self) -> None:
        """Initialize the HTTP client."""
        self.session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
        self.lock = asyncio.Lock()
        self.request_in_progress = False
        self.response_json: dict[str, Any] | None = None
        self.response_text: str | None = None

    async def close(self) -> None:
        """Close the HTTP client session.

        Ensures proper cleanup of resources by closing the aiohttp session.
        """
        async with self.lock:
            await self.session.close()

    async def delete(self, url: str, headers: dict[str, Any] = None):
        """Send an HTTP DELETE request.

        Args:
            url: The URL to send the DELETE request to.
            headers (optional): HTTP headers to include in the delete request

        Returns:
            bool: True if the request was successful (status code 200), False otherwise.
        """
        try:
            async with self.session.delete(url, headers=headers) as resp:
                return resp.status == HTTPStatus.OK
        except Exception as e:
            logger.warning(f"HttpClient: error deleting {url}: {e}")
            return False

    async def send_request(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, Any],
        payload: dict[str, Any],
        call_method: CallMethod,
        http_status_codes_to_ignore: set[HTTPStatus] | None = None,
    ) -> bool:
        """Send an HTTP request with the specified parameters.

        Args:
            url: The URL to send the request to.
            params: Query parameters to include in the request.
            headers: HTTP headers to include in the request.
            payload: The request body payload.
            call_method: The HTTP method to use (POST, PUT, GET).
            http_status_codes_to_ignore: Set of HTTP status codes to treat as success.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        async with self.lock:
            self.response_json = None
            self.response_text = None
            if self.request_in_progress:
                logger.error("Request already in progress")
            self.request_in_progress = True
            try:
                if http_status_codes_to_ignore is None:
                    http_status_codes_to_ignore = set()
                data = json.dumps(payload)
                logger.info(f"HttpClient: sending request to '{url}' params={params} data={data}")

                status_code: int = -1
                method = str(call_method.value)

                async with self.session.request(method, url, data=data, headers=headers, params=params) as resp:
                    status_code = resp.status
                    content_type = resp.headers.get("Content-Type", "")

                    if "application/json" in content_type:
                        try:
                            self.response_json = await resp.json()
                        except ValueError as exc:
                            logger.warning(f"HttpClient: error parsing JSON response: {exc}")
                    else:
                        self.response_text = await resp.text()

                if status_code != HTTPStatus.OK and status_code not in http_status_codes_to_ignore:
                    logger.warning(f"HttpClient: call to '{url}' failed with response '{self.response_text}'.")
                    return False
                else:
                    return True
            except Exception as e:
                logger.warning(f"Could not connect to API {e}")
                return False
            finally:
                self.request_in_progress = False
