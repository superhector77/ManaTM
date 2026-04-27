#!/usr/bin/env python3
"""
leak_control.py
---------------
Monitors a leak sensor on GPIO4 and controls the bilge pump accordingly.
Requires: rudder_motor_pump.py in the same directory
"""

import RPi.GPIO as GPIO
import time
from rudder_motor_pump import init, set_pump, stop_all

LEAK_SENSOR_PIN = 4
POLL_INTERVAL = 0.5  # seconds between sensor checks

def main():
    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEAK_SENSOR_PIN, GPIO.IN)

    # Initialise the pHAT and pump
    init()
    print("Leak monitor started...")

    try:
        while True:
            leak_detected = GPIO.input(LEAK_SENSOR_PIN)

            if leak_detected:
                print("Leak detected! Pump ON")
                set_pump(True)
            else:
                set_pump(False)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        stop_all()
        GPIO.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()
