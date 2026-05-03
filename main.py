#!/usr/bin/env python3
"""
main.py
-------
Entry point for the ManaTM autonomous boat.
Wires together sensing, actuation, navigation, and lights.

Folder structure (all files in same directory):
    main.py
    sensing_init.py
    actuation_init.py
    Aruco.py
    qwiic_ism.py
    rudder_motor_pump.py
    boat_lights.py
    thrustangle_to_motorservo_sig.py
    leak_pump.py
    lidar_plot.py
    SLAM_Script/

Run:
    source ~/servo_env/bin/activate
    python3 main.py
"""

import asyncio
import threading
import time
import numpy as np
import sys

# ── Local imports ──────────────────────────────────────────────────────────────
from sensing_init import init_imu, init_cameras, init_leak_sensor, init_lidar
from actuation_init import (
    init_actuators, set_light_state,
    STATE_RAINBOW, STATE_LEAK, STATE_DONE
)
from Aruco import (
    process_frame, get_robot_pose,
    T_cam1_in_robot, T_cam2_in_robot
)
from thrustangle_to_motorservo_sig import thrust_angle_to_motor_servo

# ── SLAM script lives in its own subdirectory ──────────────────────────────────
sys.path.insert(0, './SLAM_Script')
from SLAM_Script.hardware_deploy import AutonomousNavigator   # your teammate's file

# ── Shared sensor buffers (written by background threads, read by main loop) ───
_lidar_buffer  = []          # list of [angle_deg, distance_m]
_lidar_lock    = threading.Lock()

_imu_buffer    = {
    "accel": (0.0, 0.0, 0.0),
    "gyro":  (0.0, 0.0, 0.0),
}
_imu_lock      = threading.Lock()

_aruco_buffer  = None        # (x, y, z, yaw_rad) or None
_aruco_lock    = threading.Lock()

_leak_detected = False
_leak_lock     = threading.Lock()

_running       = True        # set False to cleanly shut down all threads


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND SENSOR THREADS
# ══════════════════════════════════════════════════════════════════════════════

def imu_thread(imu):
    """Continuously reads IMU and updates shared buffer. ~100 Hz."""
    global _running
    print("[IMU] thread started")
    while _running:
        if imu.check_status():
            accel = imu.get_accel()
            gyro  = imu.get_gyro()
            # Convert mg → m/s² and mdps → rad/s for the navigator
            ax = accel.xData * 9.81 / 1000.0
            ay = accel.yData * 9.81 / 1000.0
            az = accel.zData * 9.81 / 1000.0
            gx = np.radians(gyro.xData / 1000.0)
            gy = np.radians(gyro.yData / 1000.0)
            gz = np.radians(gyro.zData / 1000.0)
            with _imu_lock:
                _imu_buffer["accel"] = (ax, ay, az)
                _imu_buffer["gyro"]  = (gx, gy, gz)
        time.sleep(0.01)
    print("[IMU] thread stopped")


def camera_thread(cap1, cap2):
    """Continuously reads cameras, runs ArUco, updates shared pose buffer."""
    global _running, _aruco_buffer
    print("[CAM] thread started")
    while _running:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        pose = None

        if ret1:
            pos1, rot1 = process_frame(frame1, "Cam1")
            if pos1 is not None:
                robot_pos, robot_rot = get_robot_pose(pos1, rot1, T_cam1_in_robot)
                # Extract yaw from rotation matrix
                yaw = np.arctan2(robot_rot[1, 0], robot_rot[0, 0])
                pose = (float(robot_pos[0]), float(robot_pos[1]),
                        float(robot_pos[2]), float(yaw))

        if ret2 and pose is None:   # fall back to cam2 if cam1 gave nothing
            pos2, rot2 = process_frame(frame2, "Cam2")
            if pos2 is not None:
                robot_pos, robot_rot = get_robot_pose(pos2, rot2, T_cam2_in_robot)
                yaw = np.arctan2(robot_rot[1, 0], robot_rot[0, 0])
                pose = (float(robot_pos[0]), float(robot_pos[1]),
                        float(robot_pos[2]), float(yaw))

        with _aruco_lock:
            _aruco_buffer = pose

        time.sleep(0.033)   # ~30 fps
    print("[CAM] thread stopped")


def leak_thread(leak_sensor, set_pump):
    """Polls leak sensor, drives pump, updates leak flag and light state."""
    global _running, _leak_detected
    print("[LEAK] thread started")
    while _running:
        leak = leak_sensor.is_pressed
        with _leak_lock:
            _leak_detected = leak
        set_pump(leak)
        if leak:
            set_light_state(STATE_LEAK)
        time.sleep(0.5)
    print("[LEAK] thread stopped")


async def lidar_worker(lidar):
    """Async coroutine — reads LiDAR C1 and fills shared buffer."""
    global _running, _lidar_buffer
    print("[LIDAR] worker started")
    try:
        async for point in lidar.iter_scans():
            if not _running:
                break
            if point.get('q', 0) > 0:
                angle_deg  = point['a_deg']
                dist_m     = point['d_mm'] / 1000.0
                with _lidar_lock:
                    _lidar_buffer.append([angle_deg, dist_m])
                    # Keep buffer size bounded to ~1 full rotation (360 points)
                    if len(_lidar_buffer) > 1080:
                        _lidar_buffer = _lidar_buffer[-1080:]
    except Exception as e:
        print(f"[LIDAR] error: {e}")
    print("[LIDAR] worker stopped")


def lidar_thread(lidar):
    """Runs the async lidar worker in its own event loop."""
    asyncio.run(lidar_worker(lidar))


def lights_thread(pixels, animations):
    """Runs the LED state machine on its own thread."""
    from actuation_init import run_lights
    print("[LIGHTS] thread started")
    run_lights(pixels, animations)   # blocks forever until _running=False
    print("[LIGHTS] thread stopped")


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR READ HELPERS  (called by main loop each step)
# ══════════════════════════════════════════════════════════════════════════════

def get_real_lidar_data():
    """
    Returns a list of [angle_deg, distance_m] rows matching the format
    the AutonomousNavigator expects. Falls back to empty scan if no data yet.
    """
    with _lidar_lock:
        if not _lidar_buffer:
            # Return 1080 empty rays so SLAM doesn't crash on first tick
            return [[i * (360.0 / 1080), None] for i in range(1080)]
        return list(_lidar_buffer)


def get_real_imu_data():
    """Returns (ax,ay,az) m/s², (gx,gy,gz) rad/s."""
    with _imu_lock:
        return _imu_buffer["accel"], _imu_buffer["gyro"]


def get_real_aruco_data():
    """Returns (x, y, z, yaw_rad) or None."""
    with _aruco_lock:
        return _aruco_buffer


def get_leak():
    with _leak_lock:
        return _leak_detected


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def send_motor_commands(thrust_cmd, prop_angle_cmd, set_motor, set_rudder):
    """
    Translates navigator outputs (Newtons, radians) to hardware signals
    via thrustangle_to_motorservo_sig, then sends to rudder and motor.
    Skips command if a leak is active (safety lockout).
    """
    if get_leak():
        set_motor(0.0)
        set_rudder(90)
        return

    motor_sig, servo_sig = thrust_angle_to_motor_servo(thrust_cmd, prop_angle_cmd)

    # motor_sig from translator is 0.0–1.0 (forward only per comments)
    motor_sig = float(np.clip(motor_sig, 0.0, 1.0))

    # servo_sig is already in degrees 0–180
    servo_sig = float(np.clip(servo_sig, 0.0, 180.0))

    set_motor(motor_sig)
    set_rudder(servo_sig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _running

    # ── 1. Initialise all hardware ─────────────────────────────────────────────
    print("=" * 60)
    print("  ManaTM Autonomous Boat  —  Startup")
    print("=" * 60)

    imu                                              = init_imu()
    cap1, cap2                                       = init_cameras()
    leak_sensor                                      = init_leak_sensor()
    lidar                                            = init_lidar()
    (set_rudder, set_motor, set_pump,
     pixels, animations, run_lights_fn,
     set_light_state_fn, stop_all)                  = init_actuators()

    # ── 2. Start background threads ────────────────────────────────────────────
    threads = [
        threading.Thread(target=imu_thread,    args=(imu,),              daemon=True),
        threading.Thread(target=camera_thread, args=(cap1, cap2),        daemon=True),
        threading.Thread(target=leak_thread,   args=(leak_sensor, set_pump), daemon=True),
        threading.Thread(target=lidar_thread,  args=(lidar,),            daemon=True),
        threading.Thread(target=lights_thread, args=(pixels, animations), daemon=True),
    ]
    for t in threads:
        t.start()

    # Give sensors a moment to fill their buffers before the navigator starts
    print("Warming up sensors (3s)...")
    time.sleep(3.0)

    # ── 3. Initialise navigator ────────────────────────────────────────────────
    navigator = AutonomousNavigator()
    set_light_state(STATE_RAINBOW)
    print("\nNavigator ready. Use the matplotlib UI to start missions.\n")

    # ── 4. Main control loop ───────────────────────────────────────────────────
    try:
        while True:
            # Check for active leak — halt navigation while pumping
            if get_leak():
                set_motor(0.0)
                set_rudder(90)
                time.sleep(0.1)
                continue

            # Read sensors
            lidar_data          = get_real_lidar_data()
            imu_accel, imu_gyro = get_real_imu_data()
            aruco_data          = get_real_aruco_data()

            # Run navigator step
            thrust_cmd, prop_angle_cmd = navigator.step(
                lidar_data, imu_accel, imu_gyro, aruco_data
            )

            # Send commands to hardware
            send_motor_commands(thrust_cmd, prop_angle_cmd, set_motor, set_rudder)

            # Sync light state with mission state
            if navigator.state == "MISSION_COMPLETE":
                set_light_state(STATE_DONE)
            elif navigator.state == "E_STOP":
                set_motor(0.0)
                set_rudder(90)
                set_light_state(STATE_LEAK)   # red/green flash for e-stop too

            time.sleep(0.05)   # 20 Hz control loop

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C — shutting down...")

    finally:
        _running = False
        stop_all()
        set_motor(0.0)
        set_rudder(90)
        set_pump(False)
        pixels.fill(0)
        pixels.show()
        cap1.release()
        cap2.release()
        lidar.reset()
        print("[MAIN] Clean shutdown complete.")


if __name__ == "__main__":
    main()
