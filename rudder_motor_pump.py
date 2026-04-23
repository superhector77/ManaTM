#!/usr/bin/env python3
"""
rudder_motor_pump.py
--------------------
Control functions for:
  - Rudder servo        (channel 0)
  - Brushed motor ESC   (channel 1)
  - Bilge pump          (channel 2)

Hardware: Raspberry Pi 5 + SparkFun Pi Servo pHAT (PCA9685)
Venv:     ~/servo_env  (pip install sparkfun-pi-servo-hat)

Usage:
    from rudder_motor_pump import init, set_rudder, set_motor, set_pump
    init()
    set_rudder(90)      # centre rudder
    set_motor(0.5)      # half throttle forward
    set_pump(True)      # turn bilge pump on
"""

import pi_servo_hat
import time

# ── Channel assignments ────────────────────────────────────────────────────────
RUDDER_CHANNEL = 0   # Servo (0° = hard port, 90° = centre, 180° = hard starboard)
MOTOR_CHANNEL  = 1   # Brushed ESC (speed -1.0 to +1.0, 0 = stopped)
PUMP_CHANNEL   = 2   # Bilge pump relay/MOSFET (True = on, False = off)

# ── Servo swing (degrees) ──────────────────────────────────────────────────────
SERVO_SWING = 180    # Change to 90 if using a 90-degree rudder servo

# ── ESC neutral angle (maps to 1.5ms pulse = stopped) ─────────────────────────
ESC_NEUTRAL  = 90.0
ESC_RANGE    = 90.0  # degrees either side of neutral (90 = full 1ms–2ms range)

# ── Module-level hat instance ──────────────────────────────────────────────────
_hat = None


def init():
    """
    Initialise the pHAT and arm the ESC.
    Must be called once before any other function.
    """
    global _hat
    _hat = pi_servo_hat.PiServoHat()
    _hat.restart()

    # Centre rudder
    set_rudder(90)

    # Arm ESC — hold neutral for 2 seconds so the ESC recognises the signal
    _raw_servo(MOTOR_CHANNEL, ESC_NEUTRAL)
    time.sleep(2)

    # Pump off
    set_pump(False)

    print("boat_control: initialised")


def set_rudder(angle):
    """
    Set rudder angle.

    Parameters
    ----------
    angle : float
        Rudder angle in degrees.
        0   = hard port
        90  = centred (straight ahead)
        180 = hard starboard
    """
    _check_init()
    angle = max(0.0, min(float(SERVO_SWING), float(angle)))
    _hat.move_servo_position(RUDDER_CHANNEL, angle, SERVO_SWING)


def set_motor(speed):
    """
    Set motor speed.

    Parameters
    ----------
    speed : float
        Motor speed from -1.0 (full reverse) to +1.0 (full forward).
        0.0 = stopped.
    """
    _check_init()
    speed = max(-1.0, min(1.0, float(speed)))

    # Map -1.0–+1.0 to ESC angle range (0–180 degrees)
    # 0.0 maps to ESC_NEUTRAL (90°), ±1.0 maps to 0° or 180°
    angle = ESC_NEUTRAL + (speed * ESC_RANGE)
    _raw_servo(MOTOR_CHANNEL, angle)


def set_pump(on):
    """
    Turn the bilge pump on or off.

    Parameters
    ----------
    on : bool
        True  = pump on
        False = pump off
    """
    _check_init()
    angle = 180.0 if on else 0.0
    _raw_servo(PUMP_CHANNEL, angle)


def stop_all():
    """
    Safe shutdown — centres rudder, stops motor, turns pump off.
    Call this in an exception handler or on program exit.
    """
    _check_init()
    set_rudder(90)
    set_motor(0.0)
    set_pump(False)
    print("boat_control: all stopped")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _raw_servo(channel, angle):
    """Send a raw angle to a channel, clamped to 0–180."""
    angle = max(0.0, min(180.0, float(angle)))
    _hat.move_servo_position(channel, angle, 180)


def _check_init():
    """Raise an error if init() has not been called."""
    if _hat is None:
        raise RuntimeError("boat_control: call init() before using control functions")


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising...")
    init()

    print("Rudder: hard port")
    set_rudder(0)
    time.sleep(1)

    print("Rudder: centre")
    set_rudder(90)
    time.sleep(1)

    print("Rudder: hard starboard")
    set_rudder(180)
    time.sleep(1)

    print("Rudder: centre")
    set_rudder(90)
    time.sleep(1)

    print("Motor: half forward")
    set_motor(0.5)
    time.sleep(2)

    print("Motor: stop")
    set_motor(0.0)
    time.sleep(1)

    print("Motor: half reverse")
    set_motor(-0.5)
    time.sleep(2)

    print("Motor: stop")
    set_motor(0.0)
    time.sleep(1)

    print("Pump: on")
    set_pump(True)
    time.sleep(2)

    print("Pump: off")
    set_pump(False)

    stop_all()
    print("Test complete")
