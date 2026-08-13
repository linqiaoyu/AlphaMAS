#!/usr/bin/env python3
"""Generate the deterministic, company-free PNG used by the M1 Qwen smoke test."""

from __future__ import annotations

import argparse
import hashlib
import struct
import zlib
from pathlib import Path

WIDTH = 640
HEIGHT = 400


def _set_pixel(pixels: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def _line(
    pixels: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_pixel(pixels, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _rect(
    pixels: bytearray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    for y in range(max(0, top), min(HEIGHT, bottom + 1)):
        for x in range(max(0, left), min(WIDTH, right + 1)):
            _set_pixel(pixels, x, y, color)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def build_png() -> bytes:
    pixels = bytearray([250, 250, 248] * WIDTH * HEIGHT)
    grid = (220, 224, 226)
    axis = (55, 62, 68)
    for y in (50, 100, 150, 200, 250, 300):
        _line(pixels, 50, y, 610, y, grid)
    for x in range(50, 611, 70):
        _line(pixels, x, 30, x, 350, grid)
    _line(pixels, 50, 30, 50, 350, axis)
    _line(pixels, 50, 350, 610, 350, axis)

    close = 100
    for index in range(36):
        drift = (index % 7) - 2 + (1 if index > 17 else 0)
        open_value = close + ((index * 5) % 7) - 3
        close = open_value + drift
        high = max(open_value, close) + 3 + (index % 3)
        low = min(open_value, close) - 3 - ((index + 1) % 3)
        x = 68 + index * 15
        scale = 2
        y_open = 285 - open_value * scale
        y_close = 285 - close * scale
        y_high = 285 - high * scale
        y_low = 285 - low * scale
        up = close >= open_value
        color = (38, 139, 87) if up else (190, 55, 55)
        _line(pixels, x, y_high, x, y_low, color)
        _rect(pixels, x - 4, min(y_open, y_close), x + 4, max(y_open, y_close), color)
        volume = 16 + ((index * 11) % 38)
        _rect(pixels, x - 4, 348 - volume, x + 4, 348, color)

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * WIDTH * 3 : (row + 1) * WIDTH * 3])
        for row in range(HEIGHT)
    )
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _chunk(b"IEND", b"")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = build_png()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"{hashlib.sha256(payload).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
