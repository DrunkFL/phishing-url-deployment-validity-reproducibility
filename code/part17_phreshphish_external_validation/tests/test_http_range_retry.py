from __future__ import annotations

import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from http_parquet import HTTPRangeReader  # noqa: E402


PAYLOAD = b"abcdef"


class RangeHandler(BaseHTTPRequestHandler):
    request_count = 0
    transient_failures = 0
    permanent_status = None

    def do_GET(self) -> None:
        type(self).request_count += 1
        if type(self).permanent_status is not None:
            self.send_response(type(self).permanent_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if type(self).transient_failures > 0:
            type(self).transient_failures -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        value = self.headers.get("Range", "")
        if not value.startswith("bytes="):
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start_text, end_text = value.removeprefix("bytes=").split("-", maxsplit=1)
        start = int(start_text)
        end = min(int(end_text), len(PAYLOAD) - 1)
        body = PAYLOAD[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class LocalRangeServer:
    def __enter__(self) -> str:
        RangeHandler.request_count = 0
        RangeHandler.transient_failures = 0
        RangeHandler.permanent_status = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}/fixture.parquet"

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class HTTPRangeRetryTests(unittest.TestCase):
    def test_transient_status_is_retried_then_range_read_succeeds(self) -> None:
        with LocalRangeServer() as url:
            RangeHandler.transient_failures = 1
            with HTTPRangeReader(url, block_size=3, max_retries=2, backoff_factor=0) as reader:
                self.assertEqual(reader.read(), PAYLOAD)
            self.assertGreaterEqual(RangeHandler.request_count, 3)

    def test_permanent_status_is_not_retried(self) -> None:
        with LocalRangeServer() as url:
            RangeHandler.permanent_status = 404
            with self.assertRaises(requests.HTTPError):
                HTTPRangeReader(url, max_retries=3, backoff_factor=0)
            self.assertEqual(RangeHandler.request_count, 1)


if __name__ == "__main__":
    unittest.main()
