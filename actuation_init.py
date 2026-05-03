#!/usr/bin/env python3
"""
actuation_init.py
-----------------
Initializes all actuators on the ManaTM boat:
    - Rudder servo      (channel 0, via SparkFun Servo pHAT)
    - Motor ESC         (channel 1, via SparkFun Servo pHAT)
    - Bilge pump        (channel 2, via SparkFun Servo pHAT)
    - NeoPixel LEDs     (GPIO18 + GPIO21, 12 pixels, LED state machine)

Usage:
    from actuation_init import init_actuators
    set_rudder, set_motor, set_pump, run_lights, stop_all = init_actuators()
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
from rudder_motor_pump import (
    init as _hat_init,
    set_rudder,
    set_motor,
    set_pump,
    stop_all
)

# ── Hardware config ─────────────────────────────────────────────────────────────
NEOPIXEL        = board.D18
NEOPIXEL_2      = board.D21
NUM_PIXELS      = 12
FLASH_INTERVAL  = 0.3   # seconds between red/blue flashes

# ── States ──────────────────────────────────────────────────────────────────────
STATE_RAINBOW = "RAINBOW"
STATE_LEAK    = "LEAK"
STATE_DONE    = "DONE"


# ── NeoPixel setup ──────────────────────────────────────────────────────────────
class Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    def __init__(self, pin, pin2, size, **kwargs):
        self._pin  = pin
        self._pin2 = pin2
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin,  buf)
        neopixel_write(self._pin2, buf)


def _init_pixels():
    pixels = Pi5Pixelbuf(NEOPIXEL, NEOPIXEL_2, NUM_PIXELS,
                         auto_write=True, byteorder="BGR")
    animations = AnimationSequence(
        Rainbow(pixels,        speed=0.02, period=2),
        RainbowChase(pixels,   speed=0.02, size=5, spacing=3),
        RainbowComet(pixels,   speed=0.02, tail_length=7, bounce=True),
        RainbowSparkle(pixels, speed=0.02, num_sparkles=15),
        advance_interval=5,
        auto_clear=True,
    )
    return pixels, animations


# ── LED state machine ───────────────────────────────────────────────────────────
_current_state = STATE_RAINBOW
_state_lock    = threading.Lock()


def set_light_state(new_state):
    global _current_state
    with _state_lock:
        if _current_state != new_state:
            print(f"Light state: {_current_state} -> {new_state}")
            _current_state = new_state


def get_light_state():
    with _state_lock:
        return _current_state


def run_lights(pixels, animations):
    """
    Run the LED state machine loop. Call this on the main thread
    or in a dedicated thread. Reads state set by set_light_state().

    States:
        RAINBOW  - cycles through rainbow animations (default)
        LEAK     - alternates red/green flash
        DONE     - solid blue
    """
    flash_color = RED
    last_flash  = time.monotonic()

    while True:
        state = get_light_state()

        if state == STATE_RAINBOW:
            animations.animate()

        elif state == STATE_LEAK:
            now = time.monotonic()
            if now - last_flash >= FLASH_INTERVAL:
                flash_color = RED if flash_color == GREEN else GREEN
                pixels.fill(flash_color)
                pixels.show()
                last_flash = now

        elif state == STATE_DONE:
            pixels.fill(BLUE)
            pixels.show()
            time.sleep(0.1)


def init_actuators():
    """
    Initialize all actuators.

    Returns:
        set_rudder      - set_rudder(angle)   0=port, 90=centre, 180=starboard
        set_motor       - set_motor(speed)    -1.0 to +1.0
        set_pump        - set_pump(bool)      True=on, False=off
        pixels          - raw pixel buffer for direct color control
        run_lights      - run_lights(pixels, animations) — call in main/thread
        set_light_state - set_light_state(STATE_RAINBOW | STATE_LEAK | STATE_DONE)
        stop_all        - stop_all() safe shutdown of rudder/motor/pump
    """
    print("Initialising servo pHAT (rudder, motor, pump)...")
    _hat_init()   # arms ESC — takes ~2 seconds
    print("Servo pHAT ready")

    print("Initialising NeoPixels...")
    pixels, animations = _init_pixels()
    pixels.fill(0)
    print("NeoPixels ready")

    print("\nAll actuators initialised\n")
    return set_rudder, set_motor, set_pump, pixels, animations, run_lights, set_light_state, stop_all


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rudder, motor, pump, pixels, animations, lights, set_state, stop = init_actuators()

    print("Testing rudder...")
    rudder(0);   time.sleep(0.5)
    rudder(90);  time.sleep(0.5)
    rudder(180); time.sleep(0.5)
    rudder(90)

    print("Testing motor (brief)...")
    motor(0.2);  time.sleep(1)
    motor(0.0)

    print("Testing pump...")
    pump(True);  time.sleep(1)
    pump(False)

    print("Testing LEDs — rainbow (3s)...")
    start = time.monotonic()
    while time.monotonic() - start < 3:
        lights(pixels, animations)

    print("Testing LEDs — leak flash (3s)...")
    set_state(STATE_LEAK)
    start = time.monotonic()
    while time.monotonic() - start < 3:
        lights(pixels, animations)

    print("Testing LEDs — done/blue (2s)...")
    set_state(STATE_DONE)
    start = time.monotonic()
    while time.monotonic() - start < 2:
        lights(pixels, animations)

    stop()
    pixels.fill(0)
    pixels.show()
    time.sleep(0.02)
    print("Actuation check complete")
