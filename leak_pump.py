#!/usr/bin/env python3
"""
leak_control.py
---------------
Monitors a leak sensor on GPIO4 and controls the bilge pump accordingly.
Requires: rudder_motor_pump.py in the same directory
"""

from gpiozero import Button
import time
from rudder_motor_pump import init, set_pump, stop_all

LEAK_SENSOR_PIN = 4
POLL_INTERVAL = 0.5  # seconds between sensor checks

def main():
    leak_sensor = Button(LEAK_SENSOR_PIN, pull_up=False)

    init()
    print("Leak monitor started...")

    try:
        while True:
            if leak_sensor.is_pressed:
                print("Leak detected! Pump ON")
                set_pump(True)
            else:
                set_pump(False)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        stop_all()
        print("Done.")

if __name__ == "__main__":
    main()
