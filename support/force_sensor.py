#!/usr/bin/env python3
"""
Kunwei KWR75D six-axis force/torque sensor driver over RS422 (USB-FTDI).

Protocol (from KWR75-RS422 datasheet):
  - Baudrate: 460800, 8N1, no flow control
  - Start command: 0x48 0xAA 0x0D 0x0A  (sensor begins streaming at ~1 kHz)
  - Stop command:  0x43 0xAA 0x0D 0x0A
  - Data frame: 28 bytes
      [0:2]   header  0x48 0xAA
      [2:6]   Fx      float32 little-endian (unit: Kg)
      [6:10]  Fy      float32 little-endian (unit: Kg)
      [10:14] Fz      float32 little-endian (unit: Kg)
      [14:18] Mx      float32 little-endian (unit: Kg·m)
      [18:22] My      float32 little-endian (unit: Kg·m)
      [22:26] Mz      float32 little-endian (unit: Kg·m)
      [26:28] footer  0x0D 0x0A
  - Multiply raw values by 9.81 to convert to N / N·m

Zero/tare: no hardware command — software offset (average N frames at startup).
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial

if __package__ in (None, ""):
    from pathlib import Path
    import sys

    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from utils.runtime_config import (
    DEFAULT_FORCE_SENSOR_FALLBACK_PORT,
    DEFAULT_FORCE_SENSOR_PORT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = DEFAULT_FORCE_SENSOR_PORT
FALLBACK_PORT = DEFAULT_FORCE_SENSOR_FALLBACK_PORT
BAUDRATE = 460800
TIMEOUT_S = 0.5

FRAME_SIZE = 28
HEADER = bytes([0x48, 0xAA])
FOOTER = bytes([0x0D, 0x0A])
START_CMD = bytes([0x48, 0xAA, 0x0D, 0x0A])
STOP_CMD = bytes([0x43, 0xAA, 0x0D, 0x0A])

GRAVITY = 9.81

DEFAULT_TARE_FRAMES = 100  # frames to average for software zero


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FTReading:
    """Six-axis force/torque reading in SI units (N, N·m)."""
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    timestamp: float  # time.monotonic()

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.fx, self.fy, self.fz, self.mx, self.my, self.mz)

    def __repr__(self) -> str:
        return (
            f"FT(Fx={self.fx:+8.3f}N  Fy={self.fy:+8.3f}N  Fz={self.fz:+8.3f}N  "
            f"Mx={self.mx:+8.5f}Nm  My={self.my:+8.5f}Nm  Mz={self.mz:+8.5f}Nm)"
        )


# ---------------------------------------------------------------------------
# Sensor driver
# ---------------------------------------------------------------------------

class ForceSensor:
    """Kunwei KWR75D force/torque sensor driver.

    Usage:
        sensor = ForceSensor()
        sensor.start()         # open serial, send start command, begin reading
        sensor.tare()          # software zero (blocks ~0.1s)
        reading = sensor.get() # latest FTReading (thread-safe)
        sensor.stop()          # stop streaming and close serial
    """

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        fallback_port: str = FALLBACK_PORT,
        baudrate: int = BAUDRATE,
    ) -> None:
        self._port = port
        self._fallback_port = fallback_port
        self._baudrate = baudrate

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._latest: Optional[FTReading] = None

        # Software tare offset (subtracted from raw)
        self._offset_fx = 0.0
        self._offset_fy = 0.0
        self._offset_fz = 0.0
        self._offset_mx = 0.0
        self._offset_my = 0.0
        self._offset_mz = 0.0

        # Tare accumulation
        self._tare_accumulator: Optional[list[tuple[float, ...]]] = None
        self._tare_event: Optional[threading.Event] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open serial port, send start command, launch reader thread."""
        if self._running:
            return

        port = self._port
        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=TIMEOUT_S,
            )
        except (serial.SerialException, OSError):
            port = self._fallback_port
            self._ser = serial.Serial(
                port=port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=TIMEOUT_S,
            )

        # Flush any stale data
        self._ser.reset_input_buffer()
        self._ser.write(START_CMD)

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print(f"[ForceSensor] started on {port} @ {self._baudrate}")

    def stop(self) -> None:
        """Stop streaming and close serial."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(STOP_CMD)
                time.sleep(0.05)
            except Exception:
                pass
            self._ser.close()
            self._ser = None
        print("[ForceSensor] stopped")

    def get(self) -> Optional[FTReading]:
        """Return the latest tare-compensated reading, or None if not yet available."""
        with self._lock:
            return self._latest

    def tare(self, num_frames: int = DEFAULT_TARE_FRAMES) -> None:
        """Software zero: average num_frames readings as offset. Blocks until done."""
        if not self._running:
            raise RuntimeError("sensor is not running; call start() first")

        self._tare_event = threading.Event()
        self._tare_accumulator = []
        self._tare_target = num_frames

        # Wait for reader thread to collect enough frames
        if not self._tare_event.wait(timeout=max(2.0, num_frames * 0.005)):
            print(f"[ForceSensor] tare timeout (got {len(self._tare_accumulator or [])}/{num_frames} frames)")
            self._tare_accumulator = None
            self._tare_event = None
            return

        samples = self._tare_accumulator
        self._tare_accumulator = None
        self._tare_event = None

        if not samples:
            print("[ForceSensor] tare failed: no samples collected")
            return

        n = len(samples)
        self._offset_fx = sum(s[0] for s in samples) / n
        self._offset_fy = sum(s[1] for s in samples) / n
        self._offset_fz = sum(s[2] for s in samples) / n
        self._offset_mx = sum(s[3] for s in samples) / n
        self._offset_my = sum(s[4] for s in samples) / n
        self._offset_mz = sum(s[5] for s in samples) / n
        print(
            f"[ForceSensor] tare done ({n} frames): "
            f"offset F=({self._offset_fx:.3f}, {self._offset_fy:.3f}, {self._offset_fz:.3f}) N  "
            f"M=({self._offset_mx:.5f}, {self._offset_my:.5f}, {self._offset_mz:.5f}) Nm"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_frame(frame: bytes) -> tuple[float, float, float, float, float, float]:
        """Parse 28-byte frame → (Fx, Fy, Fz, Mx, My, Mz) in N and N·m."""
        fx, fy, fz, mx, my, mz = struct.unpack_from('<6f', frame, 2)
        fx *= GRAVITY
        fy *= GRAVITY
        fz *= GRAVITY
        mx *= GRAVITY
        my *= GRAVITY
        mz *= GRAVITY
        return fx, fy, fz, mx, my, mz

    def _reader_loop(self) -> None:
        """Background thread: read serial, find frames, update latest reading."""
        buf = bytearray()
        ser = self._ser
        assert ser is not None

        while self._running:
            try:
                waiting = ser.in_waiting
                chunk = ser.read(max(waiting, 1))
            except (serial.SerialException, OSError):
                if self._running:
                    time.sleep(0.01)
                continue

            if not chunk:
                continue

            buf.extend(chunk)

            # Extract complete frames
            while len(buf) >= FRAME_SIZE:
                idx = buf.find(HEADER)
                if idx == -1:
                    buf.clear()
                    break
                if idx > 0:
                    del buf[:idx]  # discard bytes before header
                if len(buf) < FRAME_SIZE:
                    break

                # Check footer
                if buf[FRAME_SIZE - 2 : FRAME_SIZE] != FOOTER:
                    del buf[:1]  # bad frame, skip one byte
                    continue

                frame = bytes(buf[:FRAME_SIZE])
                del buf[:FRAME_SIZE]

                raw = self._parse_frame(frame)
                now = time.monotonic()

                # Tare accumulation (uses raw, before offset)
                acc = self._tare_accumulator
                if acc is not None:
                    acc.append(raw)
                    if len(acc) >= self._tare_target:
                        evt = self._tare_event
                        if evt is not None:
                            evt.set()

                # Apply offset
                reading = FTReading(
                    fx=raw[0] - self._offset_fx,
                    fy=raw[1] - self._offset_fy,
                    fz=raw[2] - self._offset_fz,
                    mx=raw[3] - self._offset_mx,
                    my=raw[4] - self._offset_my,
                    mz=raw[5] - self._offset_mz,
                    timestamp=now,
                )

                with self._lock:
                    self._latest = reading

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KWR75D force sensor test")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--tare", action="store_true", help="Tare on startup")
    parser.add_argument("--hz", type=float, default=10.0, help="Print rate (Hz)")
    args = parser.parse_args()

    sensor = ForceSensor(port=args.port)
    sensor.start()
    time.sleep(0.5)  # wait for a few frames

    if args.tare:
        print("Taring...")
        sensor.tare()

    dt = 1.0 / args.hz
    try:
        while True:
            r = sensor.get()
            if r is not None:
                print(f"\r{r}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print()
    finally:
        sensor.stop()
