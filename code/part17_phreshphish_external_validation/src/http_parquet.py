from __future__ import annotations

import io
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)
RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)
DEFAULT_BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_FACTOR = 1.0


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reader with a single read-ahead block.

    PyArrow uses seeks to read Parquet metadata and selected column chunks. The
    reader rejects servers that ignore Range so a URL-only run cannot silently
    download the HTML column or an entire shard.
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: int = DEFAULT_READ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=RETRYABLE_STATUS_CODES,
            backoff_factor=backoff_factor,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        response = self.session.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise OSError(f"Server did not honor HTTP Range for {url}")
        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if not match:
            raise OSError(f"Missing or invalid Content-Range for {url}")
        response_start, response_end, total_length = map(int, match.groups())
        if (response_start, response_end) != (0, 0) or len(response.content) != 1:
            raise OSError(f"Invalid HTTP Range probe response for {url}")
        self.length = total_length

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise OSError("Negative seek position")
        self.position = min(position, self.length)
        return self.position

    def _fetch(self, start: int, minimum_size: int) -> None:
        fetch_size = max(self.block_size, minimum_size)
        end = min(self.length - 1, start + fetch_size - 1)
        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise OSError(f"Server stopped honoring HTTP Range for {self.url}")
        match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if not match:
            raise OSError(f"Missing or invalid Content-Range for {self.url}")
        response_start, response_end, total_length = map(int, match.groups())
        if (response_start, response_end, total_length) != (start, end, self.length):
            raise OSError(f"Unexpected Content-Range for {self.url}")
        if len(response.content) != end - start + 1:
            raise OSError(f"Truncated HTTP Range response for {self.url}")
        self.cache_start = start
        self.cache = response.content

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        output = bytearray()
        while len(output) < size:
            cache_end = self.cache_start + len(self.cache)
            if not (self.cache_start <= self.position < cache_end):
                self._fetch(self.position, size - len(output))
                cache_end = self.cache_start + len(self.cache)
            available = min(size - len(output), cache_end - self.position)
            start = self.position - self.cache_start
            output.extend(self.cache[start : start + available])
            self.position += available
        return bytes(output)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.session.close()
        super().close()
