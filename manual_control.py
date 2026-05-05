#!/usr/bin/env python3
"""
manual_control.py
-----------------
Interactive terminal control for rudder and motor via SSH.

  UP arrow    = motor forward at 30% (hold to keep running, releases to stop)
  DOWN arrow  = motor reverse at 30% (hold to keep running, releases to stop)
  LEFT arrow  = rudder hard port 0°  (hold to keep, releases to centre)
  RIGHT arrow = rudder hard starboard 180° (hold to keep, releases to centre)
  P           = toggle bilge pump
  Q / Ctrl+C  = quit

Usage:
    ~/servo_env/bin/python3 ~/manual_control.py
"""

import sys
import tty
import termios
import threading
import time
from rudder_motor_pump import init, set_rudder, set_motor, set_pump, stop_all

# ── Settings ───────────────────────────────────────────────────────────────────
MOTOR_SPEED  = 0.3   # speed when arrow is held (0.0 to 1.0)
KEY_TIMEOUT  = 0.15  # seconds after last keypress before treating as released

# ── State ──────────────────────────────────────────────────────────────────────
motor_speed  = 0.0
rudder_angle = 90.0
pump_on      = False

# Timestamps of last keypress for each arrow
_last_up    = 0.0
_last_down  = 0.0
_last_left  = 0.0
_last_right = 0.0
_running    = True


def read_key():
    """Read a single keypress including arrow keys (3-byte escape sequences)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def print_status():
    """Print current state on a single refreshing line."""
    pump_str = "ON " if pump_on else "OFF"

    if motor_speed > 0:
        motor_str = f"FWD {int(motor_speed * 100):3d}%"
    elif motor_speed < 0:
        motor_str = f"REV {int(abs(motor_speed) * 100):3d}%"
    else:
        motor_str = "STOP    "

    if rudder_angle < 45:
        rudder_str = "PORT     "
    elif rudder_angle > 135:
        rudder_str = "STARBOARD"
    else:
        rudder_str = "CENTRE   "

    sys.stdout.write(
        f"\r  Motor: {motor_str}   Rudder: {rudder_str} {rudder_angle:5.1f}   Pump: {pump_str}   "
    )
    sys.stdout.flush()


def monitor_loop():
    """
    Background thread — checks every KEY_TIMEOUT seconds whether a held key
    has been released (i.e. no recent keypress) and resets motor/rudder to
    neutral if so.
    """
    global motor_speed, rudder_angle, _running

    while _running:
        now = time.time()
        changed = False

        # Motor: up or down arrow held?
        motor_held = (now - _last_up < KEY_TIMEOUT) or (now - _last_down < KEY_TIMEOUT)
        if not motor_held and motor_speed != 0.0:
            motor_speed = 0.0
            set_motor(0.0)
            changed = True

        # Rudder: left or right arrow held?
        rudder_held = (now - _last_left < KEY_TIMEOUT) or (now - _last_right < KEY_TIMEOUT)
        if not rudder_held and rudder_angle != 90.0:
            rudder_angle = 90.0
            set_rudder(90.0)
            changed = True

        if changed:
            print_status()

        time.sleep(0.05)


def main():
    global motor_speed, rudder_angle, pump_on
    global _last_up, _last_down, _last_left, _last_right, _running

    print("Initialising...")
    init()

    print("\n── Manual Control ──────────────────────────────────────────")
    print("  UP         Motor forward 30% (release to stop)")
    print("  DOWN       Motor reverse 30% (release to stop)")
    print("  LEFT       Rudder hard port (release to centre)")
    print("  RIGHT      Rudder hard starboard (release to centre)")
    print("  P          Toggle bilge pump")
    print("  Q          Quit")
    print("────────────────────────────────────────────────────────────\n")
    print_status()

    # Start background monitor thread
    monitor = threading.Thread(target=monitor_loop, daemon=True)
    monitor.start()

    try:
        while True:
            key = read_key()
            now = time.time()

            if key == '\x1b[A':          # UP arrow — forward
                _last_up = now
                if motor_speed != MOTOR_SPEED:
                    motor_speed = MOTOR_SPEED
                    set_motor(motor_speed)

            elif key == '\x1b[B':        # DOWN arrow — reverse
                _last_down = now
                if motor_speed != -MOTOR_SPEED:
                    motor_speed = -MOTOR_SPEED
                    set_motor(motor_speed)

            elif key == '\x1b[D':        # LEFT arrow — hard port
                _last_left = now
                if rudder_angle != 0.0:
                    rudder_angle = 0.0
                    set_rudder(0.0)

            elif key == '\x1b[C':        # RIGHT arrow — hard starboard
                _last_right = now
                if rudder_angle != 180.0:
                    rudder_angle = 180.0
                    set_rudder(180.0)

            elif key.lower() == 'p':     # P — toggle pump
                pump_on = not pump_on
                set_pump(pump_on)

            elif key.lower() == 'q' or key == '\x03':  # Q or Ctrl+C
                break

            print_status()

    finally:
        _running = False
        print("\n\nShutting down...")
        stop_all()
        print("Done.")


if __name__ == "__main__":
    main()
