#!/usr/bin/env python3
"""Offline tests: PYTHONPATH=/path/to/fpga-scheduler/src python3 this_file.py.

The mock HTTP daemon exercises the real SDK lease/stream behavior. No board
connection is made and all test credentials are temporary synthetic values.
"""
import base64
from contextlib import redirect_stdout
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

SPEC = importlib.util.spec_from_file_location("scheduler_runner", Path(__file__).with_name("run-fpga-scheduler.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
UART = '@@FPGA_START {"version":1}\nverify: PASS\n@@FPGA_RESULT {"version":1,"status":"exited","app_exit_code":0}\n'


def finished(ident="0" * 32, uart=UART):
    return dict(id=ident, state="completed", created=0,
                result=dict(output=uart, uart_output=uart, node="mock", seconds=0.1,
                            execution_status="exited", app_exit_code=0, exit_code=0))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle_request(self):
        server = self.server
        if self.headers.get("Authorization") != "Bearer offline-test-token":
            return self.reply({"error": "unauthorized"}, 401)
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        path = self.path.split("?")[0]
        if path == "/health":
            return self.reply({"ok": True})
        if path == "/devices":
            return self.reply({"devices": [{"node": "mock", "number": 0, "available": True}], "errors": {}})
        if path.startswith("/sessions/"):
            if self.command == "PUT":
                server.opens += 1
                return self.reply(dict(heartbeat_seconds=0.03, lease_seconds=0.3))
            if self.command == "POST":
                server.heartbeats += 1
            if self.command == "DELETE":
                server.closes += 1
            return self.reply({})
        if path.endswith("/image"):
            ident = path.split("/")[2]
            server.images[ident] = data
            return self.reply(dict(id=ident, sha256=hashlib.sha256(data).hexdigest()))
        if path == "/jobs" and self.command == "POST":
            payload = json.loads(data)
            ident = payload["id"]
            if server.active is not None or not self.headers.get("X-FPGA-Session"):
                return self.reply({"error": "overlapping jobs or missing session"}, 409)
            server.active = ident
            server.jobs[ident] = dict(id=ident, state="running", created=time.time())
            server.started[ident] = time.monotonic()
            return self.reply(server.jobs[ident], 202)
        if path.endswith("/output"):
            ident = path.split("/")[2]
            done = time.monotonic() - server.started[ident] > 0.25
            uart = b""
            if done:
                kind = server.images[ident]
                task = finished(ident, "PASS\n" if kind == b"pass-only" else UART)
                if kind == b"exit-seven":
                    task["state"] = "failed"
                    task["result"].update(app_exit_code=7, exit_code=1,
                                          uart_output=UART.replace('"app_exit_code":0', '"app_exit_code":7'))
                server.jobs[ident] = task
                server.active = None
                uart = task["result"]["uart_output"].encode()
            return self.reply(dict(id=ident, offset=0, next_offset=len(uart),
                                   data=base64.b64encode(uart).decode(), done=done, task=server.jobs[ident]))
        if path.startswith("/jobs/"):
            return self.reply(server.jobs[path.split("/")[2]])
        return self.reply({"error": "unexpected request"}, 404)

    def reply(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = do_PUT = do_DELETE = handle_request


class SchedulerRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.token = self.root / "client.token"
        self.token.write_text("offline-test-token\n")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.opens = self.server.closes = self.server.heartbeats = 0
        self.server.images, self.server.jobs, self.server.started = {}, {}, {}
        self.server.active = None
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_images(self, kinds, *extra):
        images = []
        for index, kind in enumerate(kinds):
            image = self.root / f"{index}.bin"
            image.write_bytes(kind)
            images.append(str(image))
        with redirect_stdout(io.StringIO()):
            code = RUNNER.main(["--url", self.url, "--token-file", str(self.token), "--no-proxy",
                                "--output", str(self.root / "report"), *extra, *images])
        summary = json.loads((self.root / "report/summary.json").read_text())
        return code, summary

    def test_serial_jobs_keep_leases_and_save_full_results(self):
        code, summary = self.run_images([b"success", b"success"])
        self.assertEqual(code, 0)
        self.assertTrue(summary["ok"])
        self.assertEqual(len(summary["tasks"]), 2)
        self.assertEqual(self.server.opens, 1)
        self.assertEqual(self.server.closes, 1)
        self.assertGreater(self.server.heartbeats, 2)
        for record in summary["tasks"]:
            evidence = json.loads((self.root / "report" / record["evidence"]).read_text())
            self.assertEqual(evidence["task"]["result"]["uart_output"], UART)

    def test_pass_text_and_nonzero_exit_are_not_success(self):
        code, summary = self.run_images([b"pass-only", b"exit-seven", b"success"], "--keep-going")
        self.assertEqual(code, 1)
        self.assertFalse(summary["ok"])
        self.assertEqual([record["ok"] for record in summary["tasks"]], [False, False, True])

    def test_failure_stops_before_next_upload_by_default(self):
        code, summary = self.run_images([b"exit-seven", b"success"])
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["tasks"]), 1)
        self.assertEqual(len(self.server.images), 1)

    def test_health_does_not_open_session_or_submit(self):
        with redirect_stdout(io.StringIO()):
            code = RUNNER.main(["--url", self.url, "--token-file", str(self.token), "--no-proxy",
                                "--health", "--output", str(self.root / "health")])
        self.assertEqual(code, 0)
        self.assertEqual(self.server.opens, 0)
        self.assertEqual(self.server.images, {})

    def test_protocol_and_api_must_agree(self):
        self.assertTrue(RUNNER.verdict(finished())["ok"])
        for text in ("PASS\n", UART + UART, UART.rstrip("\n"),
                     UART.replace('"app_exit_code":0', '"app_exit_code":true'),
                     UART.replace('"version":1', '"version":2')):
            with self.subTest(uart=text):
                self.assertFalse(RUNNER.verdict(finished(uart=text))["ok"])
        for key, value in (("app_exit_code", 7), ("exit_code", 1), ("execution_status", "timeout"),
                           ("app_exit_code", False), ("uart_output", None)):
            task = copy.deepcopy(finished())
            task["result"][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(RUNNER.verdict(task)["ok"])


if __name__ == "__main__":
    unittest.main()
