"""
E-Stop (Emergency Stop) System for Robot Tugboat
=================================================
Provides hardware-level and software-level emergency stop functionality.
Supports multiple trigger sources: physical button, software command,
watchdog timeout, network loss, and sensor fault.

Usage:
    estop = EStop(config=EStopConfig())
    estop.register_stop_callback(my_shutdown_fn)
    estop.start()

    # Trigger manually:
    estop.trigger(EStopReason.SOFTWARE_COMMAND, "Operator requested stop")

    # Release (after inspection):
    estop.release()
"""

import time
import threading
import logging
import enum
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from collections import deque

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("estop")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EStopConfig:
    """Tuneable parameters for the e-stop system."""

    # Watchdog: if heartbeat() is not called within this period, trigger e-stop
    watchdog_timeout_s: float = 2.0

    # How often the watchdog checker loop runs (should be << watchdog_timeout_s)
    watchdog_poll_interval_s: float = 0.1

    # Maximum log entries retained in the event log
    event_log_max_entries: int = 200

    # If True, require an explicit release() call before motors can restart
    latching: bool = True

    # Optional GPIO pin number for physical e-stop button (set None to disable)
    # Integration stub — wire your GPIO library here.
    hw_button_gpio_pin: Optional[int] = None  # e.g. 17


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------
class EStopReason(enum.Enum):
    HARDWARE_BUTTON = "hardware_button"
    SOFTWARE_COMMAND = "software_command"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    NETWORK_LOSS = "network_loss"
    SENSOR_FAULT = "sensor_fault"
    OVER_SPEED = "over_speed"
    LOW_BATTERY = "low_battery"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class EStopState(enum.Enum):
    NORMAL = "NORMAL"        # Operating normally
    STOPPING = "STOPPING"    # Stop commands being sent, motors decelerating
    STOPPED = "STOPPED"      # Motors off; waiting for release
    FAULTED = "FAULTED"      # Hard fault — requires inspection + manual reset


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------
@dataclass
class EStopEvent:
    timestamp: float
    state: EStopState
    reason: Optional[EStopReason]
    message: str

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        r = self.reason.value if self.reason else "—"
        return f"[{t}] {self.state.name:<10}  reason={r:<20}  {self.message}"


# ---------------------------------------------------------------------------
# Core E-Stop class
# ---------------------------------------------------------------------------
class EStop:
    """
    Thread-safe emergency stop controller for a robot tugboat.

    Lifecycle
    ---------
    1. Instantiate with config.
    2. Register one or more stop/resume callbacks.
    3. Call start() to begin background watchdog thread.
    4. Call heartbeat() periodically from your main control loop.
    5. Call trigger() from any thread or interrupt handler when needed.
    6. After resolving the fault, call release() to return to NORMAL.
    7. Call stop() to shut down the watchdog thread cleanly.
    """

    def __init__(self, config: EStopConfig = EStopConfig()) -> None:
        self.config = config
        self._state = EStopState.NORMAL
        self._lock = threading.Lock()

        self._stop_callbacks: List[Callable[[EStopReason, str], None]] = []
        self._resume_callbacks: List[Callable[[], None]] = []

        self._event_log: deque[EStopEvent] = deque(maxlen=config.event_log_max_entries)
        self._last_heartbeat = time.monotonic()
        self._active_reason: Optional[EStopReason] = None

        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background watchdog thread."""
        if self._running:
            log.warning("EStop already started.")
            return
        self._running = True
        self._last_heartbeat = time.monotonic()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="estop-watchdog"
        )
        self._watchdog_thread.start()
        log.info("EStop watchdog started (timeout=%.1fs).", self.config.watchdog_timeout_s)

        if self.config.hw_button_gpio_pin is not None:
            self._init_hw_button()

    def stop(self) -> None:
        """Stop the background watchdog thread (does NOT trigger e-stop)."""
        self._running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
        log.info("EStop watchdog stopped.")

    def heartbeat(self) -> None:
        """
        Call this regularly from the main control loop to feed the watchdog.
        If not called within watchdog_timeout_s, an e-stop is triggered.
        """
        if self._state == EStopState.NORMAL:
            self._last_heartbeat = time.monotonic()

    def trigger(
        self,
        reason: EStopReason = EStopReason.UNKNOWN,
        message: str = "",
    ) -> None:
        """
        Trigger an emergency stop from any thread.

        Safe to call multiple times — subsequent calls while already stopped
        are logged but do not re-fire callbacks.
        """
        with self._lock:
            if self._state in (EStopState.STOPPED, EStopState.FAULTED, EStopState.STOPPING):
                log.debug("EStop already active; ignoring duplicate trigger (%s).", reason)
                return

            self._state = EStopState.STOPPING
            self._active_reason = reason
            event_msg = message or f"E-stop triggered: {reason.value}"
            self._log_event(EStopState.STOPPING, reason, event_msg)

        log.critical("🚨 E-STOP TRIGGERED — reason=%s  %s", reason.value, message)
        self._send_stop_commands(reason, message)

        with self._lock:
            self._state = EStopState.STOPPED
            self._log_event(EStopState.STOPPED, reason, "Motors halted.")

        log.critical("🛑 E-STOP ACTIVE — all propulsion halted.")

    def release(self) -> bool:
        """
        Attempt to release the e-stop and return to NORMAL operation.

        Returns True on success, False if the state cannot be cleared
        (e.g., still in FAULTED state requiring manual intervention).
        """
        with self._lock:
            if self._state == EStopState.NORMAL:
                log.info("EStop already in NORMAL state.")
                return True

            if self._state == EStopState.FAULTED:
                log.error("Cannot release: system is FAULTED. Manual reset required.")
                return False

            if self._state == EStopState.STOPPING:
                log.warning("Cannot release: stop sequence still in progress.")
                return False

            self._state = EStopState.NORMAL
            self._active_reason = None
            self._last_heartbeat = time.monotonic()  # Reset watchdog
            self._log_event(EStopState.NORMAL, None, "E-stop released. Resuming normal ops.")

        log.info("✅ E-stop released — system returning to NORMAL.")
        self._fire_resume_callbacks()
        return True

    def fault(self, message: str = "Hard fault detected.") -> None:
        """
        Transition to FAULTED state. Requires manual physical reset to clear.
        Use for severe conditions (hull breach sensor, propeller jam, etc.).
        """
        with self._lock:
            self._state = EStopState.FAULTED
            self._log_event(EStopState.FAULTED, EStopReason.SENSOR_FAULT, message)
        log.critical("⚠️  SYSTEM FAULTED: %s", message)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_stop_callback(
        self, fn: Callable[[EStopReason, str], None]
    ) -> None:
        """Register a function called when e-stop fires. Signature: fn(reason, message)."""
        self._stop_callbacks.append(fn)

    def register_resume_callback(self, fn: Callable[[], None]) -> None:
        """Register a function called when e-stop is released. Signature: fn()."""
        self._resume_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Convenience trigger helpers
    # ------------------------------------------------------------------

    def trigger_network_loss(self) -> None:
        self.trigger(EStopReason.NETWORK_LOSS, "Communication link lost.")

    def trigger_sensor_fault(self, sensor_name: str) -> None:
        self.trigger(EStopReason.SENSOR_FAULT, f"Sensor fault: {sensor_name}")

    def trigger_over_speed(self, speed_knots: float) -> None:
        self.trigger(EStopReason.OVER_SPEED, f"Over-speed: {speed_knots:.1f} kn")

    def trigger_low_battery(self, voltage: float) -> None:
        self.trigger(EStopReason.LOW_BATTERY, f"Battery critical: {voltage:.2f}V")

    # ------------------------------------------------------------------
    # Status / diagnostics
    # ------------------------------------------------------------------

    @property
    def state(self) -> EStopState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state != EStopState.NORMAL

    @property
    def active_reason(self) -> Optional[EStopReason]:
        return self._active_reason

    def get_event_log(self) -> List[EStopEvent]:
        return list(self._event_log)

    def print_event_log(self) -> None:
        print("\n=== EStop Event Log ===")
        for ev in self._event_log:
            print(ev)
        print("=======================\n")

    def status_report(self) -> str:
        elapsed = time.monotonic() - self._last_heartbeat
        return (
            f"State       : {self._state.name}\n"
            f"Active reason: {self._active_reason}\n"
            f"Watchdog age: {elapsed:.2f}s / {self.config.watchdog_timeout_s}s\n"
            f"Events logged: {len(self._event_log)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        """Background thread: monitors heartbeat and fires e-stop on timeout."""
        while self._running:
            time.sleep(self.config.watchdog_poll_interval_s)
            if self._state != EStopState.NORMAL:
                continue
            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self.config.watchdog_timeout_s:
                log.warning(
                    "Watchdog timeout: no heartbeat for %.2fs (limit %.1fs).",
                    elapsed,
                    self.config.watchdog_timeout_s,
                )
                self.trigger(
                    EStopReason.WATCHDOG_TIMEOUT,
                    f"No heartbeat for {elapsed:.2f}s.",
                )

    def _send_stop_commands(self, reason: EStopReason, message: str) -> None:
        """
        Issue motor stop commands and invoke registered callbacks.
        In production: write 0 to all thruster/motor channels here
        before firing user callbacks.
        """
        # --- Hardware interface stub ---
        # self.motor_driver.set_all_throttle(0)
        # self.motor_driver.enable_brakes()
        log.info("Motor stop commands issued (hardware stub).")

        for cb in self._stop_callbacks:
            try:
                cb(reason, message)
            except Exception as exc:  # noqa: BLE001
                log.error("Stop callback raised an exception: %s", exc)

    def _fire_resume_callbacks(self) -> None:
        for cb in self._resume_callbacks:
            try:
                cb()
            except Exception as exc:  # noqa: BLE001
                log.error("Resume callback raised an exception: %s", exc)

    def _log_event(
        self,
        state: EStopState,
        reason: Optional[EStopReason],
        message: str,
    ) -> None:
        ev = EStopEvent(
            timestamp=time.time(),
            state=state,
            reason=reason,
            message=message,
        )
        self._event_log.append(ev)

    def _init_hw_button(self) -> None:
        """
        Stub for physical GPIO button integration.
        Replace with your GPIO library (RPi.GPIO, gpiozero, etc.).
        """
        pin = self.config.hw_button_gpio_pin
        log.info("HW e-stop button configured on GPIO pin %d (stub).", pin)
        # Example with RPi.GPIO:
        # import RPi.GPIO as GPIO
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        # GPIO.add_event_detect(
        #     pin, GPIO.FALLING, callback=self._hw_button_isr, bouncetime=50
        # )

    def _hw_button_isr(self, channel: int) -> None:
        """Interrupt service routine for physical e-stop button."""
        log.critical("Physical e-stop button pressed on GPIO %d.", channel)
        self.trigger(EStopReason.HARDWARE_BUTTON, f"GPIO {channel} asserted.")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Robot Tugboat E-Stop Demo ===\n")

    # --- Callbacks ---
    def on_stop(reason: EStopReason, message: str) -> None:
        print(f"  [CALLBACK] STOP fired — {reason.value}: {message}")

    def on_resume() -> None:
        print("  [CALLBACK] System resumed — all clear.")

    # --- Setup ---
    config = EStopConfig(watchdog_timeout_s=2.0, latching=True)
    estop = EStop(config=config)
    estop.register_stop_callback(on_stop)
    estop.register_resume_callback(on_resume)
    estop.start()

    # --- Scenario 1: Watchdog timeout ---
    print("Scenario 1: Watchdog timeout (no heartbeat for 3s)")
    time.sleep(3.0)  # Deliberately skip heartbeat
    print(estop.status_report())
    estop.release()
    time.sleep(0.2)

    # --- Scenario 2: Software command ---
    print("\nScenario 2: Software e-stop command")
    estop.heartbeat()
    time.sleep(0.1)
    estop.trigger(EStopReason.SOFTWARE_COMMAND, "Obstacle detected by LIDAR.")
    print(estop.status_report())
    estop.release()
    time.sleep(0.2)

    # --- Scenario 3: Sensor fault ---
    print("\nScenario 3: Sensor fault")
    estop.heartbeat()
    time.sleep(0.1)
    estop.trigger_sensor_fault("hull_pressure_sensor")
    estop.release()
    time.sleep(0.2)

    # --- Scenario 4: Normal heartbeat keeps watchdog happy ---
    print("\nScenario 4: Healthy heartbeat for 2s (no e-stop expected)")
    for _ in range(20):
        estop.heartbeat()
        time.sleep(0.1)
    print("Still running:", estop.state.name)

    estop.stop()
    estop.print_event_log()
    print("Demo complete.")
