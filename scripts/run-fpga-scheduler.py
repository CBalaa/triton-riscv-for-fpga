#!/usr/bin/env python3
"""Run local .bin files serially, retaining UART and strict scheduler evidence.

Install the fpga-scheduler SDK or pass --sdk-root /path/to/fpga-scheduler.
Credentials are read only from --token-file / FPGA_SCHEDULER_TOKEN_FILE.
--dry-run never opens a connection; --health only queries health and devices.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener, install_opener


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def image_manifest(paths):
    images = []
    seen = set()
    for name in paths:
        path = Path(name).expanduser().resolve()
        candidates = sorted(path.rglob("*.bin")) if path.is_dir() else [path]
        if not candidates:
            raise ValueError(f"No .bin files found in {path}")
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            if candidate.suffix != ".bin" or not candidate.is_file():
                raise ValueError(f"Expected a .bin file: {candidate}")
            data = candidate.read_bytes()
            if not 0 < len(data) <= 16 * 1024 * 1024:
                raise ValueError(f"Image must be 1 byte to 16 MiB: {candidate}")
            seen.add(candidate)
            images.append(dict(path=str(candidate), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
    return images


def verdict(task):
    """Require both scheduler completion and independently parsed UART v1."""
    from fpga_scheduler.remote.runner import Completion, ProtocolError
    errors = []
    result = task.get("result") or {}
    if task.get("state") != "completed":
        errors.append("task state is not completed")
    if result.get("execution_status") != "exited":
        errors.append("execution_status is not exited")
    for key in ("app_exit_code", "exit_code"):
        if type(result.get(key)) is not int or result[key] != 0:
            errors.append(f"{key} is not integer zero")
    uart = result.get("uart_output")
    protocol = Completion()
    if not isinstance(uart, str):
        errors.append("missing independent uart_output")
    else:
        try:
            protocol.feed(uart.encode("utf-8"))
            protocol.finish()
            if not protocol.started or protocol.result is None:
                errors.append("missing UART START or RESULT record")
            elif protocol.result != dict(version=1, status="exited", app_exit_code=0):
                errors.append("UART terminal record does not report successful exit")
        except ProtocolError as error:
            errors.append(str(error))
    return dict(ok=not errors, errors=errors, uart_result=protocol.result)


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help=".bin files or directories searched recursively")
    parser.add_argument("--url", "--endpoint", default=os.environ.get("FPGA_SCHEDULER_URL"))
    parser.add_argument("--token-file", default=os.environ.get("FPGA_SCHEDULER_TOKEN_FILE"))
    parser.add_argument("--sdk-root", default=os.environ.get("FPGA_SCHEDULER_SDK_ROOT"),
                        help="SDK checkout directory (or its src directory)")
    parser.add_argument("--output", type=Path, help="new report directory (must not already exist)")
    parser.add_argument("--timeout", type=int, default=600, help="load plus execution budget, 1..600 seconds")
    parser.add_argument("--queue-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true", help="write image manifest without network/SDK/token")
    parser.add_argument("--health", action="store_true", help="query service/device status without submitting")
    parser.add_argument("--no-proxy", action="store_true", help="ignore HTTP proxy environment for this process")
    parser.add_argument("--keep-going", action="store_true", help="continue after a completed failed task")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 600 or not 1 <= args.queue_timeout <= 86400 or args.request_timeout <= 0:
        parser.error("invalid timeout: execution 1..600, queue 1..86400, request > 0")
    if args.health and (args.images or args.dry_run):
        parser.error("--health is read-only and cannot be combined with images or --dry-run")
    if not args.health and not args.images:
        parser.error("provide at least one .bin file/directory, or use --health")
    if not args.dry_run:
        if not args.url or not args.token_file:
            parser.error("--url and --token-file (or their environment variables) are required")
        url = urlsplit(args.url)
        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            parser.error("URL must be HTTP(S), without credentials, query or fragment")
    return args


def main(argv=None):
    args = arguments(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or Path("build/fpga-results") / (stamp + "-" + uuid.uuid4().hex[:8])).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Cannot create report directory: {error}", file=sys.stderr)
        return 1
    summary = dict(created_utc=stamp, mode="dry-run" if args.dry_run else "health" if args.health else "run",
                   endpoint=args.url, execution="not_started", ok=None, images=[], tasks=[])
    status = 1
    try:
        summary["images"] = image_manifest(args.images)
        if args.dry_run:
            summary["execution"] = "not_run"
            status = 0
            return status
        if args.sdk_root:
            root = Path(args.sdk_root).expanduser().resolve()
            sys.path.insert(0, str(root / "src" if (root / "src/fpga_scheduler").is_dir() else root))
        try:
            from fpga_scheduler import Client
        except ImportError as error:
            raise RuntimeError("Install fpga-scheduler or provide --sdk-root") from error
        if args.no_proxy:
            install_opener(build_opener(ProxyHandler({})))
        with Client(args.url, token_file=args.token_file, request_timeout=args.request_timeout) as client:
            summary["health"] = client.health()
            summary["devices"] = client.devices()
            if args.health:
                summary["execution"] = "not_run"
                summary["ok"] = True
                status = 0
                return status
            summary["execution"] = "running"
            for index, image in enumerate(summary["images"], 1):
                record = dict(image=image, ok=False, task=None)
                summary["tasks"].append(record)
                evidence = output / f"{index:03d}-{Path(image['path']).stem}.json"
                record["evidence"] = evidence.name
                try:
                    if hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest() != image["sha256"]:
                        raise RuntimeError("Image changed after manifest creation")
                    task = client.submit(image["path"], timeout=args.timeout, queue_timeout=args.queue_timeout)
                    record["task"] = asdict(task)
                    save_json(evidence, record)
                    print(f"[{index}/{len(summary['images'])}] {image['path']} task={task.id}", flush=True)
                    with client.stream_output(task.id, timeout=args.queue_timeout + args.timeout + 300) as stream:
                        for chunk in stream:
                            print(chunk, end="", flush=True)
                        record["task"] = asdict(stream.task)
                    record["validation"] = verdict(record["task"])
                    record["ok"] = record["validation"]["ok"]
                except BaseException as error:
                    record["error"] = str(error) or type(error).__name__
                    if getattr(error, "details", None):
                        record["error_details"] = error.details
                    ident = getattr(error, "task_id", None) or (record["task"] or {}).get("id")
                    if ident:
                        record["task_id"] = ident
                        try:
                            record["task"] = asdict(client.job(ident))
                        except Exception:
                            pass
                    raise
                finally:
                    save_json(evidence, record)
                    save_json(output / "summary.json", summary)
                print(f"\nverified={record['ok']} evidence={evidence}", flush=True)
                if not record["ok"] and not args.keep_going:
                    break
            summary["execution"] = "finished"
            summary["ok"] = len(summary["tasks"]) == len(summary["images"]) and all(t["ok"] for t in summary["tasks"])
            status = 0 if summary["ok"] else 1
    except KeyboardInterrupt:
        summary.update(execution="interrupted", ok=False, error="Interrupted; client cancels unfinished jobs")
        status = 130
    except Exception as error:
        summary.update(execution="error", ok=False, error=str(error))
        print(f"error: {error}", file=sys.stderr)
    finally:
        save_json(output / "summary.json", summary)
        print(f"Report: {output / 'summary.json'}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
