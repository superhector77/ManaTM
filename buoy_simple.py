#!/usr/bin/env python3
import time
import math
import threading
import asyncio
import numpy as np
import cv2

# Hardware Imports
from rplidarc1 import RPLidar
from Aruco import process_frame # We only need this to check if a tag exists
from rudder_motor_pump import init as hat_init, set_rudder, set_motor, stop_all
from thrustangle_to_motorservo_sig import thrust_angle_to_motor_servo

# --- Configuration ---
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUD = 460800
CAM_INDEX = 0

CRUISE_THRUST = 10.0
TARGET_RADIUS = 1.0   # Meters away from buoy

# PID gains for orbit
KP_DIST = 25.0
KP_ANG = 0.5

# Lap Counting Config
TARGET_LAPS = 3
LAP_COOLDOWN = 4.0    # Seconds to ignore the camera after spotting a tag

# --- Shared Data ---
_lidar_buffer = []
_lidar_lock = threading.Lock()

_tag_in_view = False
_aruco_lock = threading.Lock()

_running = True

# ==========================================
# BACKGROUND THREADS
# ==========================================
async def _lidar_queue_reader(lidar):
    global _running, _lidar_buffer
    while _running:
        if lidar.output_queue.qsize() < 1:
            await asyncio.sleep(0.01)
            continue
        point = await lidar.output_queue.get()
        if point.get('q', 0) > 0:
            angle_deg, dist_m = point['a_deg'], point['d_mm'] / 1000.0
            with _lidar_lock:
                _lidar_buffer.append((angle_deg, dist_m))
                if len(_lidar_buffer) > 1080:
                    _lidar_buffer = _lidar_buffer[-1080:]

def lidar_thread_func():
    lidar = RPLidar(port=LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=0.2)
    async def run():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(lidar.simple_scan(make_return_dict=True))
            tg.create_task(_lidar_queue_reader(lidar))
    asyncio.run(run())

def camera_thread_func():
    global _running, _tag_in_view
    print("Initializing Camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while _running:
        ret, frame = cap.read()
        if ret:
            # Run your existing Aruco function
            pos, rot = process_frame(frame, "Cam1")
            
            with _aruco_lock:
                # If pos is not None, we see a tag
                _tag_in_view = (pos is not None)
                
        time.sleep(0.033) # Limit to ~30 FPS
    cap.release()

# ==========================================
# HELPERS
# ==========================================
def get_nearest_object():
    with _lidar_lock:
        if not _lidar_buffer: return None, None
        closest = min(_lidar_buffer, key=lambda x: x[1])
        return closest[0], closest[1]

def send_commands(thrust, steering_angle_deg):
    motor_sig, servo_sig = thrust_angle_to_motor_servo(thrust, math.radians(steering_angle_deg))
    set_motor(float(np.clip(motor_sig, 0.0, 1.0)))
    set_rudder(float(np.clip(servo_sig, 0.0, 180.0)))

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    global _running
    print("Initializing Hardware...")
    hat_init()
    
    threading.Thread(target=lidar_thread_func, daemon=True).start()
    threading.Thread(target=camera_thread_func, daemon=True).start()
    
    print("Warming up sensors (3s)...")
    time.sleep(3.0)

    state = "LEAVE_DOCK"
    lap_count = 0
    last_tag_time = 0.0

    try:
        while True:
            angle_deg, dist_m = get_nearest_object()
            if dist_m is None:
                time.sleep(0.05)
                continue

            # Wrap angle to -180 to 180
            if angle_deg > 180: angle_deg -= 360

            if state == "LEAVE_DOCK":
                send_commands(CRUISE_THRUST, 0.0)
                if dist_m < 3.0 and abs(angle_deg) < 30.0:
                    print(f"Buoy detected at {dist_m:.2f}m. Starting Orbit.")
                    state = "ORBIT"
                    # Reset lap timer so we can immediately detect the first tag
                    last_tag_time = 0.0 

            elif state == "ORBIT":
                # --- LiDAR Steering Logic ---
                dist_err = dist_m - TARGET_RADIUS
                ang_err = angle_deg - (-90.0)
                rudder_cmd = np.clip((KP_DIST * dist_err) + (KP_ANG * ang_err), -45.0, 45.0)
                send_commands(CRUISE_THRUST, rudder_cmd)

                # --- ArUco Lap Counting Logic ---
                with _aruco_lock:
                    seeing_tag_now = _tag_in_view
                
                current_time = time.time()
                
                if seeing_tag_now and (current_time - last_tag_time > LAP_COOLDOWN):
                    lap_count += 1
                    print(f"[!] ArUco Tag Spotted! Lap {lap_count}/{TARGET_LAPS} marked.")
                    last_tag_time = current_time
                    
                if lap_count >= TARGET_LAPS:
                    print("Target laps completed.")
                    state = "STOP"

            elif state == "STOP":
                send_commands(0.0, 0.0)
                print("Mission Complete.")
                break

            time.sleep(0.05) # 20Hz Loop

    except KeyboardInterrupt:
        print("\nManual Override.")
    finally:
        _running = False
        time.sleep(0.5)
        send_commands(0.0, 0.0)
        stop_all()
        print("Shutdown safe.")

if __name__ == "__main__":
    main()