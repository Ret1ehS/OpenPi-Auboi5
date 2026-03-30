#!/usr/bin/env python3
"""
Jetson-side gripper controller for the Lebai gripper over USB-RS485.

The gripper command path remains:
- Jetson serial -> USB-RS485 -> gripper

Tool IO is only used to keep the tool-side power rail enabled before serial
communication. It is not used as the open/close command channel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import serial


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = SCRIPT_DIR

DEFAULT_SDK_ROOT = "/home/orin/openpi/aubo_sdk/aubo_sdk-0.24.1-rc.3-Linux_aarch64+318754d"
DEFAULT_TOOL_IO_HELPER_CPP = str(SUPPORT_DIR / "tool_io_helper.cpp")
DEFAULT_TOOL_IO_HELPER_BIN = "/home/orin/openpi/scripts/.build/tool_io_helper"

DEFAULT_ROBOT_IP = "192.168.1.100"
DEFAULT_PORT_RPC = 30004
DEFAULT_USER = "aubo"
DEFAULT_PASSWORD = "123456"
DEFAULT_TOOL_POWER_VOLTAGE = 24

DEFAULT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AB0LIYTU-if00-port0"
FALLBACK_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 0.2

SLAVE_ID = 0x01

REG_POSITION = 0x9C45
REG_DONE = 0x9C47
REG_UNHOMED = 0x9C49
REG_TARGET = 0x9C40
REG_FORCE = 0x9C41
REG_SPEED = 0x9C4A

TARGET_OPEN = 100
TARGET_CLOSE = 0

OPENPI_OPEN = 1
OPENPI_CLOSE = 0
OPENPI_THRESHOLD = 0.6
DEFAULT_SPEED = 20
DEFAULT_FORCE = 5
DEFAULT_POLL_INTERVAL_S = 0.05
DEFAULT_WAIT_TIMEOUT_S = 2.5
DEFAULT_STABLE_READS = 3
DEFAULT_POSITION_TOL = 5


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _tool_power_voltage() -> int:
    return _env_int("OPENPI_TOOL_VOLTAGE", DEFAULT_TOOL_POWER_VOLTAGE)


def _companion_cpp_path() -> Path:
    return Path(DEFAULT_TOOL_IO_HELPER_CPP).resolve()


def build_tool_io_helper(
    sdk_root: str = DEFAULT_SDK_ROOT,
    helper_cpp: str | None = None,
    helper_bin: str = DEFAULT_TOOL_IO_HELPER_BIN,
) -> Path:
    sdk = Path(sdk_root).resolve()
    src = Path(helper_cpp).resolve() if helper_cpp else _companion_cpp_path()
    out = Path(helper_bin).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    compile_cmd = [
        "g++",
        "-std=c++17",
        str(src),
        f"-I{sdk / 'include'}",
        f"-L{sdk / 'lib'}",
        f"-Wl,-rpath,{sdk / 'lib'}",
        "-laubo_sdk",
        "-lpthread",
        "-o",
        str(out),
    ]
    subprocess.run(compile_cmd, check=True)
    return out


def _run_tool_io_helper(
    *,
    helper_bin: str = DEFAULT_TOOL_IO_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT_RPC,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    set_voltage: int | None = None,
    status: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(Path(helper_bin).resolve()),
        "--robot-ip",
        robot_ip,
        "--port",
        str(port),
        "--user",
        user,
        "--password",
        password,
    ]
    if set_voltage is not None:
        cmd.extend(["--set-voltage", str(int(set_voltage))])
    if status:
        cmd.append("--status")
    return subprocess.run(cmd, capture_output=True, text=True)


def ensure_tool_power_enabled(
    *,
    helper_bin: str = DEFAULT_TOOL_IO_HELPER_BIN,
    robot_ip: str = DEFAULT_ROBOT_IP,
    port: int = DEFAULT_PORT_RPC,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    voltage: int | None = None,
) -> None:
    build_tool_io_helper(helper_bin=helper_bin)
    completed = _run_tool_io_helper(
        helper_bin=helper_bin,
        robot_ip=robot_ip,
        port=port,
        user=user,
        password=password,
        set_voltage=_tool_power_voltage() if voltage is None else voltage,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"tool power enable failed rc={completed.returncode}: "
            f"stdout={completed.stdout[:200]!r} stderr={completed.stderr[:200]!r}"
        )


def crc16_modbus(data: bytes) -> bytes:
    crc = 0xFFFF
    for ch in data:
        crc ^= ch
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def build_read_holding_register(register: int) -> bytes:
    req = bytes([SLAVE_ID, 0x03, (register >> 8) & 0xFF, register & 0xFF, 0x00, 0x01])
    return req + crc16_modbus(req)


def build_write_single_register_via_0x10(register: int, value: int) -> bytes:
    payload = bytes(
        [
            SLAVE_ID,
            0x10,
            (register >> 8) & 0xFF,
            register & 0xFF,
            0x00,
            0x01,
            0x02,
            (value >> 8) & 0xFF,
            value & 0xFF,
        ]
    )
    return payload + crc16_modbus(payload)


def parse_read_response(buf: bytes) -> int | None:
    for i in range(0, max(0, len(buf) - 6)):
        frame = buf[i : i + 7]
        if len(frame) < 7:
            continue
        if frame[0] != SLAVE_ID or frame[1] != 0x03 or frame[2] != 0x02:
            continue
        if crc16_modbus(frame[:-2]) != frame[-2:]:
            continue
        return (frame[3] << 8) | frame[4]
    return None


def parse_write_echo(buf: bytes, register: int) -> bool:
    if len(buf) < 8:
        return False
    frame = buf[:8]
    if frame[0] != SLAVE_ID or frame[1] != 0x10:
        return False
    if frame[2] != ((register >> 8) & 0xFF) or frame[3] != (register & 0xFF):
        return False
    if frame[4] != 0x00 or frame[5] != 0x01:
        return False
    return crc16_modbus(frame[:-2]) == frame[-2:]


@dataclass
class GripperStatus:
    position: int | None
    done: int | None
    unhomed: int | None


class GripperController:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.ser: serial.Serial | None = None

    def open(self) -> None:
        ensure_tool_power_enabled()
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout_s,
        )
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self) -> "GripperController":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> serial.Serial:
        if self.ser is None:
            raise RuntimeError("serial port is not open")
        return self.ser

    def _request(self, frame: bytes, read_len: int = 64, settle_s: float = 0.04) -> bytes:
        ser = self._ensure_open()
        ser.reset_input_buffer()
        ser.write(frame)
        ser.flush()
        time.sleep(settle_s)
        return ser.read(read_len)

    def read_register(self, register: int, tries: int = 4) -> int | None:
        req = build_read_holding_register(register)
        for _ in range(tries):
            rsp = self._request(req)
            value = parse_read_response(rsp)
            if value is not None:
                return value
            time.sleep(0.08)
        return None

    def write_register(self, register: int, value: int, tries: int = 3) -> bool:
        frame = build_write_single_register_via_0x10(register, value)
        for _ in range(tries):
            rsp = self._request(frame)
            if parse_write_echo(rsp, register):
                return True
            time.sleep(0.10)
        return False

    def read_status(self) -> GripperStatus:
        return GripperStatus(
            position=self.read_register(REG_POSITION),
            done=self.read_register(REG_DONE),
            unhomed=self.read_register(REG_UNHOMED),
        )

    def set_target(self, target: int, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> bool:
        ok_speed = self.write_register(REG_SPEED, speed)
        ok_force = self.write_register(REG_FORCE, force)
        ok_target = self.write_register(REG_TARGET, target)
        return ok_speed and ok_force and ok_target

    def set_openpi_state(self, state01: int, speed: int = DEFAULT_SPEED, force: int = DEFAULT_FORCE) -> bool:
        if state01 not in (OPENPI_OPEN, OPENPI_CLOSE):
            raise ValueError("OpenPI gripper state must be 0 or 1")
        target = TARGET_OPEN if state01 == OPENPI_OPEN else TARGET_CLOSE
        return self.set_target(target=target, speed=speed, force=force)

    def apply_openpi_action(
        self,
        raw_action: float,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
    ) -> bool:
        state01 = OPENPI_OPEN if float(raw_action) > OPENPI_THRESHOLD else OPENPI_CLOSE
        return self.set_openpi_state(state01=state01, speed=speed, force=force)


def resolve_port(requested: str | None) -> str:
    if requested:
        return requested
    try:
        ensure_tool_power_enabled()
        with serial.Serial(DEFAULT_PORT):
            return DEFAULT_PORT
    except Exception:
        return FALLBACK_PORT


def gripper_status_to_openpi_state(
    status: GripperStatus,
    *,
    position_tol: int = DEFAULT_POSITION_TOL,
) -> int | None:
    if status.position is None:
        return None
    if int(status.position) >= TARGET_OPEN - int(position_tol):
        return OPENPI_OPEN
    if int(status.position) <= TARGET_CLOSE + int(position_tol):
        return OPENPI_CLOSE
    return None


def is_gripper_at_state(
    target_state: int,
    status: GripperStatus,
    *,
    position_tol: int = DEFAULT_POSITION_TOL,
) -> bool:
    if target_state not in (OPENPI_OPEN, OPENPI_CLOSE):
        raise ValueError("target_state must be 0 or 1")
    if status.position is None:
        return False

    pos = int(status.position)
    if target_state == OPENPI_OPEN:
        return pos >= TARGET_OPEN - int(position_tol)
    return pos <= TARGET_CLOSE + int(position_tol)


def is_gripper_stably_closed(
    status: GripperStatus,
    *,
    reference_position: int | None,
    stable_count: int,
    stable_reads: int = DEFAULT_STABLE_READS,
    position_tol: int = DEFAULT_POSITION_TOL,
) -> tuple[bool, int, int | None]:
    if status.unhomed not in (0, None) and int(status.unhomed) != 0:
        return False, 0, None
    if status.position is None:
        return False, 0, None

    pos = int(status.position)
    moved_from_open = pos < TARGET_OPEN - int(position_tol)
    if not moved_from_open:
        return False, 0, pos

    if reference_position is None:
        return False, 1, pos

    if abs(pos - int(reference_position)) <= int(position_tol):
        next_count = stable_count + 1
        return next_count >= int(stable_reads), next_count, pos

    return False, 1, pos


def _wait_for_target_state(
    ctrl: GripperController,
    target_state: int,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    stable_reads: int = DEFAULT_STABLE_READS,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    position_tol: int = DEFAULT_POSITION_TOL,
) -> bool:
    stable_count = 0
    contact_stable_count = 0
    contact_position: int | None = None
    deadline = time.monotonic() + max(0.0, timeout_s)

    while time.monotonic() <= deadline:
        status = ctrl.read_status()
        if is_gripper_at_state(target_state, status, position_tol=position_tol):
            stable_count += 1
            if stable_count >= int(stable_reads):
                return True
        else:
            stable_count = 0

        if target_state == OPENPI_CLOSE:
            closed_on_object, contact_stable_count, contact_position = is_gripper_stably_closed(
                status,
                reference_position=contact_position,
                stable_count=contact_stable_count,
                stable_reads=stable_reads,
                position_tol=position_tol,
            )
            if closed_on_object:
                return True
        else:
            contact_stable_count = 0
            contact_position = None

        time.sleep(max(0.0, poll_interval_s))
    return False


def get_gripper_status(port: str | None = None) -> GripperStatus:
    with GripperController(port=resolve_port(port)) as ctrl:
        return ctrl.read_status()


def set_gripper_state(
    state01: int,
    *,
    speed: int = DEFAULT_SPEED,
    force: int = DEFAULT_FORCE,
    wait_s: float = 0.0,
    port: str | None = None,
) -> bool:
    with GripperController(port=resolve_port(port)) as ctrl:
        ok = ctrl.set_openpi_state(state01=state01, speed=speed, force=force)
        if wait_s > 0:
            time.sleep(wait_s)
        return ok


def wait_gripper_done(
    target_state: int,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    stable_reads: int = DEFAULT_STABLE_READS,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    position_tol: int = DEFAULT_POSITION_TOL,
    port: str | None = None,
) -> bool:
    with GripperController(port=resolve_port(port)) as ctrl:
        return _wait_for_target_state(
            ctrl,
            target_state,
            timeout_s=timeout_s,
            stable_reads=stable_reads,
            poll_interval_s=poll_interval_s,
            position_tol=position_tol,
        )


def set_gripper_open(
    *,
    speed: int = DEFAULT_SPEED,
    force: int = DEFAULT_FORCE,
    wait_s: float = 0.0,
    port: str | None = None,
) -> bool:
    return set_gripper_state(
        OPENPI_OPEN,
        speed=speed,
        force=force,
        wait_s=wait_s,
        port=port,
    )


def set_gripper_close(
    *,
    speed: int = DEFAULT_SPEED,
    force: int = DEFAULT_FORCE,
    wait_s: float = 0.0,
    port: str | None = None,
) -> bool:
    return set_gripper_state(
        OPENPI_CLOSE,
        speed=speed,
        force=force,
        wait_s=wait_s,
        port=port,
    )


def apply_gripper_action(
    raw_action: float,
    *,
    speed: int = DEFAULT_SPEED,
    force: int = DEFAULT_FORCE,
    wait_s: float = 0.0,
    port: str | None = None,
) -> bool:
    with GripperController(port=resolve_port(port)) as ctrl:
        ok = ctrl.apply_openpi_action(raw_action=raw_action, speed=speed, force=force)
        if wait_s > 0:
            time.sleep(wait_s)
        return ok


def command_gripper_state(
    target_state: int,
    *,
    speed: int = DEFAULT_SPEED,
    force: int = DEFAULT_FORCE,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    stable_reads: int = DEFAULT_STABLE_READS,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    position_tol: int = DEFAULT_POSITION_TOL,
    port: str | None = None,
) -> bool:
    with GripperController(port=resolve_port(port)) as ctrl:
        ok = ctrl.set_openpi_state(target_state, speed=speed, force=force)
        # The RS485 write echo is occasionally unreliable after tool-power
        # changes. Still wait on observed position so a successful motion is
        # not reported as failure just because the echo frame was missed.
        return _wait_for_target_state(
            ctrl,
            target_state,
            timeout_s=timeout_s,
            stable_reads=stable_reads,
            poll_interval_s=poll_interval_s,
            position_tol=position_tol,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the Lebai gripper over USB-RS485 with tool-IO power enabled."
    )
    parser.add_argument("--port", default=None, help="Serial port path. Defaults to FTDI by-id path.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="Gripper speed register value.")
    parser.add_argument("--force", type=int, default=DEFAULT_FORCE, help="Gripper force register value.")
    parser.add_argument("--state", type=int, choices=[0, 1], help="OpenPI-aligned binary command: 1=open, 0=close.")
    parser.add_argument("--action", type=float, help="Raw OpenPI gripper scalar.")
    parser.add_argument("--status", action="store_true", help="Read and print gripper status registers.")
    parser.add_argument("--wait", type=float, default=1.5, help="Seconds to wait before post-status read.")
    parser.add_argument("--ensure-power", action="store_true", help="Only enable tool power, do not send serial command.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    port = resolve_port(args.port)
    print(f"PORT={port}")
    print("OPENPI_MAPPING=1->open, 0->close")

    if args.ensure_power:
        ensure_tool_power_enabled()
        print(f"TOOL_POWER_ENABLED={_tool_power_voltage()}")
        return 0

    with GripperController(port=port, baudrate=args.baudrate, timeout_s=args.timeout) as ctrl:
        before = ctrl.read_status()
        print(f"STATUS_BEFORE={before}")

        if args.state is not None and args.action is not None:
            raise ValueError("Use either --state or --action, not both.")

        if args.state is not None:
            ok = ctrl.set_openpi_state(args.state, speed=args.speed, force=args.force)
            print(f"SET_STATE_OK={ok}")
        elif args.action is not None:
            mapped = OPENPI_OPEN if args.action > OPENPI_THRESHOLD else OPENPI_CLOSE
            ok = ctrl.apply_openpi_action(args.action, speed=args.speed, force=args.force)
            print(f"RAW_ACTION={args.action}")
            print(f"MAPPED_STATE={mapped}")
            print(f"SET_ACTION_OK={ok}")
        elif not args.status:
            print("No action requested. Use --status, --state, --action, or --ensure-power.")
            return 0

        if args.state is not None or args.action is not None:
            time.sleep(max(0.0, args.wait))

        after = ctrl.read_status()
        print(f"STATUS_AFTER={after}")

    return 0


__all__ = [
    "GripperController",
    "GripperStatus",
    "OPENPI_OPEN",
    "OPENPI_CLOSE",
    "OPENPI_THRESHOLD",
    "build_tool_io_helper",
    "ensure_tool_power_enabled",
    "get_gripper_status",
    "gripper_status_to_openpi_state",
    "is_gripper_at_state",
    "wait_gripper_done",
    "set_gripper_state",
    "command_gripper_state",
    "set_gripper_open",
    "set_gripper_close",
    "apply_gripper_action",
]


if __name__ == "__main__":
    raise SystemExit(main())
