"""Stdout takeover for interactive mode (redirect writes to stderr)."""

from __future__ import annotations

import asyncio
import sys
from typing import Callable

_RAW_STDOUT_RETRY_DELAY_MS = 0.01

_stdout_takeover_state: dict[str, object] | None = None
_raw_stdout_write_tail: asyncio.Future[None] | None = None
_sync_raw_stdout_write_tail: list[asyncio.Task[None]] = []


def _get_raw_stdout_write() -> Callable[[str], None]:
    if _stdout_takeover_state is not None:
        return _stdout_takeover_state["rawStdoutWrite"]  # type: ignore[return-value]
    return sys.stdout.write


def _write_raw_stdout_chunk_sync(text: str) -> None:
    write = _get_raw_stdout_write()
    while True:
        try:
            write(text)
            sys.stdout.flush()
            return
        except OSError as error:
            if getattr(error, "errno", None) not in (11, 35, 105):  # EAGAIN, EWOULDBLOCK, ENOBUFS
                raise


async def _write_raw_stdout_chunk(text: str) -> None:
    while True:
        try:
            await asyncio.to_thread(_write_raw_stdout_chunk_sync, text)
            return
        except OSError as error:
            if getattr(error, "errno", None) not in (11, 35, 105):
                raise
        await asyncio.sleep(_RAW_STDOUT_RETRY_DELAY_MS)


def take_over_stdout() -> None:
    global _stdout_takeover_state
    if _stdout_takeover_state is not None:
        return

    raw_stdout_write = sys.stdout.write
    original_stdout_write = sys.stdout.write

    def redirected_write(
        chunk: str,
        *_args: object,
        **_kwargs: object,
    ) -> int:
        sys.stderr.write(str(chunk))
        sys.stderr.flush()
        return len(str(chunk))

    sys.stdout.write = redirected_write  # type: ignore[method-assign, assignment]

    _stdout_takeover_state = {
        "rawStdoutWrite": raw_stdout_write,
        "originalStdoutWrite": original_stdout_write,
    }


def restore_stdout() -> None:
    global _stdout_takeover_state
    if _stdout_takeover_state is None:
        return
    sys.stdout.write = _stdout_takeover_state["originalStdoutWrite"]  # type: ignore[method-assign]
    _stdout_takeover_state = None


def is_stdout_taken_over() -> bool:
    return _stdout_takeover_state is not None


def write_raw_stdout(text: str) -> None:
    if not text:
        return

    async def _run() -> None:
        await _write_raw_stdout_chunk(text)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write_raw_stdout_chunk_sync(text)
        return

    global _raw_stdout_write_tail
    previous = _raw_stdout_write_tail
    if previous is None:
        task = loop.create_task(_run())
    else:

        async def _chain() -> None:
            await previous
            await _run()

        task = loop.create_task(_chain())
    _raw_stdout_write_tail = task

    def _on_done(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except Exception:
            raise SystemExit(1) from None

    task.add_done_callback(_on_done)


async def wait_for_raw_stdout_backpressure() -> None:
    global _raw_stdout_write_tail
    while True:
        tail = _raw_stdout_write_tail
        if tail is None:
            return
        await tail
        if tail is _raw_stdout_write_tail:
            return


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()
    await _write_raw_stdout_chunk("")
