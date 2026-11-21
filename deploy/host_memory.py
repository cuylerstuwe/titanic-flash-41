#!/usr/bin/env python3
"""Hold proactive compaction and optional swappiness at zero for one serving worker's lifetime."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import time

from host_state import atomic, require_volatile_directory


SYSCTL = "vm.compaction_proactiveness"
KNOB = Path("/proc/sys/vm/compaction_proactiveness")
SWAP_KNOB = Path("/proc/sys/vm/swappiness")
BOOT = Path("/proc/sys/kernel/random/boot_id")
DIRECTORY = Path("/run") / f"titanic-native-compaction-{os.getuid()}"
SCRIPT = Path(__file__).resolve()


def run(args, check=True):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=45)


def provision():
    if not DIRECTORY.exists():
        run(["sudo", "-n", "install", "-d", "-m", "0700", "-o", str(os.getuid()),
             "-g", str(os.getgid()), str(DIRECTORY)])
    require_volatile_directory(DIRECTORY)


def set_sysctl(name, knob, value):
    run(["sudo", "-n", "sysctl", "-q", "-w", f"{name}={value}"])
    if int(knob.read_text()) != value:
        raise RuntimeError(f"serving {name} setting did not take effect")


def ownership(session, acquire, zero_swappiness=False):
    require_volatile_directory(DIRECTORY)
    with (DIRECTORY / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = DIRECTORY / "owner.json"
        state = json.loads(path.read_bytes()) if path.exists() else None
        boot = BOOT.read_text().strip()
        if acquire:
            if state is not None:
                raise RuntimeError("another serving generation owns host compaction")
            knobs = [(SYSCTL, KNOB, "original")]
            if zero_swappiness:
                knobs.append(("vm.swappiness", SWAP_KNOB, "swappiness"))
            state = dict(session=session, boot=boot)
            for name, knob, field in knobs:
                state[field] = int(knob.read_text())
                if not 0 <= state[field] <= (100 if field == "original" else 200):
                    raise ValueError(f"unexpected {name} value")
            # Record every prior value before changing the first one. ExecStopPost can
            # restore a partially applied policy even if the guard is killed mid-arm.
            atomic(path, state, durable=False)
            for name, knob, field in knobs:
                if state[field]:
                    set_sysctl(name, knob, 0)
        elif state and (state["session"], state["boot"]) == (session, boot):
            # An operator can override either value independently. Attempt both
            # restorations and keep the marker if a later retry is needed.
            try:
                if state["original"] and int(KNOB.read_text()) == 0:
                    set_sysctl(SYSCTL, KNOB, state["original"])
            finally:
                if state.get("swappiness") and int(SWAP_KNOB.read_text()) == 0:
                    set_sysctl("vm.swappiness", SWAP_KNOB, state["swappiness"])
            path.unlink()


def worker(container, config):
    fmt = "{{.Id}} {{.State.Pid}} {{.State.Status}} {{json .Config.Cmd}}"
    try:
        result = run(["docker", "inspect", "--type", "container", "--format", fmt, container], check=False)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode:
        # During handoff an absent container and a temporarily unavailable Docker daemon both warrant a retry.
        return None
    identity, pid, status, arguments = result.stdout.strip().split(" ", 3)
    args = json.loads(arguments)
    if not isinstance(args, list) or not args or args[-1] != config:
        raise RuntimeError("container name belongs to a different serving generation")
    return identity, int(pid), status


def wait_for_worker(container, config):
    # Capture a pidfd while Docker still identifies the PID as this exact worker.
    # The wait then stays entirely in the kernel, including during ordinary request gaps.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        first = worker(container, config)
        if first and first[2] in ("exited", "dead", "removing"):
            return
        if first and first[1] > 0:
            try:
                fd = os.pidfd_open(first[1])
            except ProcessLookupError:
                time.sleep(.5)
                continue
            try:
                second = worker(container, config)
                if second and second[:2] == first[:2]:
                    poll = select.poll()
                    poll.register(fd, select.POLLIN)
                    poll.poll()
                    return
            finally:
                os.close(fd)
        time.sleep(.5)
    raise TimeoutError("serving worker did not appear before the startup deadline")


def watch(session, container, config, zero_swappiness=False):
    ownership(session, True, zero_swappiness)
    address = os.environ["NOTIFY_SOCKET"]
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
        notify.sendto(b"READY=1", address)
    wait_for_worker(container, config)


def control(action, session, container, config, zero_swappiness=False):
    unit = f"titanic-compaction-{session}.service"
    if action == "arm":
        provision()
        # systemd runs the matching release even if watch is killed before it can finish startup.
        run(["sudo", "-n", "systemd-run", "--quiet", "--collect", "--service-type=notify", "--unit=" + unit,
             "--property=User=" + str(os.getuid()), "--property=TimeoutStartSec=35", "--property=TimeoutStopSec=15",
             "--working-directory=" + str(SCRIPT.parent),
             "--property=ExecStopPost=/usr/bin/python3 " + SCRIPT.name + " release " + session,
             "/usr/bin/python3", str(SCRIPT), "watch", session, container, config,
             *(["--zero-swappiness"] if zero_swappiness else [])])
        held = run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0
        if not held or int(KNOB.read_text()) != 0 or (zero_swappiness and int(SWAP_KNOB.read_text()) != 0):
            raise RuntimeError("serving memory policy did not acquire this node")
    else:
        # A failed group arm can leave some ranks without a unit. Retire only this generation.
        result = run(["sudo", "-n", "systemctl", "stop", unit], check=False)
        if result.returncode not in (0, 5):
            raise RuntimeError(f"could not stop serving memory unit ({result.returncode})")
        if DIRECTORY.exists():
            ownership(session, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("arm", "retire", "watch", "release"))
    parser.add_argument("session")
    parser.add_argument("worker", nargs="*")
    parser.add_argument("--zero-swappiness", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.session):
        parser.error("expected a serving generation identifier")
    if args.action in ("arm", "watch"):
        if len(args.worker) != 2:
            parser.error("expected a container name and generation config")
        name, config = args.worker
        match = re.fullmatch(r"titanic-" + args.session + r"-r([0-3])", name)
        if match is None or config != f"/state/rank{match.group(1)}.json":
            parser.error("invalid container or generation config")
        if args.action == "watch":
            watch(args.session, name, config, args.zero_swappiness)
        else:
            control("arm", args.session, name, config, args.zero_swappiness)
    elif args.worker or args.zero_swappiness:
        parser.error("unexpected worker arguments")
    elif args.action == "release":
        ownership(args.session, False)
    else:
        control("retire", args.session, None, None)


if __name__ == "__main__":
    main()
