#!/usr/bin/env python3
"""
boat_lights.py
--------------
LED state machine with leak detection.

States:
    RAINBOW  - default, cycles through rainbow animations
    LEAK     - flashing red/blue (leak detected)
    DONE     - solid green (task complete, triggered by pressing Enter)

Requires:
    - leak_control.py in the same directory (for leak sensor)
    - pip install adafruit-led-animation adafruit-raspberry-pi5-neopixel-write
"""

import time
import threading
import board
import adafruit_pixelbuf
from adafruit_led_animation.animation.rainbow import Rainbow
from adafruit_led_animation.animation.rainbowchase import RainbowChase
from adafruit_led_animation.animation.rainbowcomet import RainbowComet
from adafruit_led_animation.animation.rainbowsparkle import RainbowSparkle
from adafruit_led_animation.sequence import AnimationSequence
from adafruit_led_animation.color import RED, BLUE, GREEN
from adafruit_raspberry_pi5_neopixel_write import neopixel_write
from gpiozero import Button
from rudder_motor_pump import init as pump_init, set_pump, stop_all

# ── Hardware config ────────────────────────────────────────────────────────────
NEOPIXEL        = board.D18
NUM_PIXELS      = 10
LEAK_SENSOR_PIN = 4
POLL_INTERVAL   = 0.5    # seconds between leak checks
FLASH_INTERVAL  = 0.3    # seconds between red/blue flashes

# ── States ─────────────────────────────────────────────────────────────────────
STATE_RAINBOW = "RAINBOW"
STATE_LEAK    = "LEAK"
STATE_DONE    = "DONE"


# ── Pixel setup (unchanged from your script) ───────────────────────────────────
class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


pixels = Pi5Pixelbuf(NEOPIXEL, NUM_PIXELS, auto_write=True, byteorder="BGR")

# ── Animations ─────────────────────────────────────────────────────────────────
rainbow         = Rainbow(pixels, speed=0.02, period=2)
rainbow_chase   = RainbowChase(pixels, speed=0.02, size=5, spacing=3)
rainbow_comet   = RainbowComet(pixels, speed=0.02, tail_length=7, bounce=True)
rainbow_sparkle = RainbowSparkle(pixels, speed=0.02, num_sparkles=15)
animations      = AnimationSequence(
    rainbow,
    rainbow_chase,
    rainbow_comet,
    rainbow_sparkle,
    advance_interval=5,
    auto_clear=True,
)

# ── Shared state ───────────────────────────────────────────────────────────────
current_state = STATE_RAINBOW
state_lock    = threading.Lock()


def set_state(new_state):
    global current_state
    with state_lock:
        if current_state != new_state:
            print(f"State: {current_state} -> {new_state}")
            current_state = new_state


def get_state():
    with state_lock:
        return current_state


# ── Leak monitor thread ────────────────────────────────────────────────────────
def leak_monitor():
    """Polls the leak sensor and updates state + pump accordingly."""
    leak_sensor = Button(LEAK_SENSOR_PIN, pull_up=False)
    print("Leak monitor started")

    while True:
        # Don't override DONE state with leak logic
        if get_state() != STATE_DONE:
            if leak_sensor.is_pressed:
                set_state(STATE_LEAK)
                set_pump(True)
            else:
                set_pump(False)
                # Only return to rainbow if we were in leak state
                if get_state() == STATE_LEAK:
                    set_state(STATE_RAINBOW)

        time.sleep(POLL_INTERVAL)


# ── Done trigger thread ────────────────────────────────────────────────────────
def done_trigger():
    """Press Enter in the terminal to trigger the DONE state."""
    print("Press Enter at any time to signal task complete")
    while True:
        input()
        set_state(STATE_DONE)
        print("Task marked complete — lights set to green")


# ── LED state machine ──────────────────────────────────────────────────────────
def run_leds():
    flash_color = RED
    last_flash  = time.monotonic()

    while True:
        state = get_state()

        if state == STATE_RAINBOW:
            animations.animate()

        elif state == STATE_LEAK:
            # Alternate red and blue flashes
            now = time.monotonic()
            if now - last_flash >= FLASH_INTERVAL:
                flash_color = BLUE if flash_color == RED else RED
                pixels.fill(flash_color)
                pixels.show()
                last_flash = now

        elif state == STATE_DONE:
            pixels.fill(GREEN)
            pixels.show()
            time.sleep(0.1)   # no need to hammer the pixels in done state


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising pump...")
    pump_init()

    # Start leak monitor in background thread
    leak_thread = threading.Thread(target=leak_monitor, daemon=True)
    leak_thread.start()

    # Start done trigger listener in background thread
    done_thread = threading.Thread(target=done_trigger, daemon=True)
    done_thread.start()

    print("Running — Ctrl+C to exit")
    try:
        run_leds()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_all()
        pixels.fill(0)
        pixels.show()
        time.sleep(0.02)
        print("Done")
