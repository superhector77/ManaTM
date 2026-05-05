#!/usr/bin/env python3
import time
import math
import threading
import asyncio
import numpy as np

# Hardware Imports
from rplidarc1 import RPLidar
from rudder_motor_pump import init as hat_init, set_rudder, set_motor, stop_all
from thrustangle_to_motorservo_sig import thrust_angle_to_motor_servo
from aruco_simple import ArucoLocator # Your class!

# --- Configuration ---
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUD = 460800

CRUISE_THRUST = 10.0
TARGET_RADIUS = 1.5   
TRANSIT_DISTANCE = 2.0 # Meters to travel in -X direction before searching

# PID gains for orbit
KP_DIST = 25.0
KP_ANG = 0.5

# --- Shared LiDAR Data ---
_lidar_buffer = []
_lidar_lock = threading.Lock()
_running = True

# ==========================================
# BACKGROUND LiDAR THREAD
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

# ==========================================
# HELPERS
# ==========================================
def get_nearest_object_on_left():
    """Returns (angle, dist) of closest object ONLY on the left side (0 to -180 deg)"""
    with _lidar_lock:
        if not _lidar_buffer: return None, None
        
        # Filter for points on the left side (angles > 180 when 0-360, or negative if wrapped)
        left_points = []
        for ang, dist in _lidar_buffer:
            wrapped_ang = ang if ang <= 180 else ang - 360
            if wrapped_ang < 0: # It is on our left
                left_points.append((wrapped_ang, dist))
                
        if not left_points: return None, None
        closest = min(left_points, key=lambda x: x[1])
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
    print("\n--- ManaTM Hybrid Initialization ---")
    hat_init()
    
    # Start LiDAR
    threading.Thread(target=lidar_thread_func, daemon=True).start()
    
    # Start ArUco
    locator = ArucoLocator()
    locator.setup()
    
    print("Warming up sensors (3s)...")
    time.sleep(3.0)

    # 1. INITIAL CALIBRATION
    print("\n[STATE: CALIBRATE] Averaging starting position...")
    start_x_samples = []
    for _ in range(10):
        x, y, yaw = locator.get_position()
        start_x_samples.append(x)
        time.sleep(0.1)
        
    start_x = sum(start_x_samples) / 10.0
    target_x = start_x - TRANSIT_DISTANCE
    print(f"-> Starting X: {start_x:.3f}m | Target X: {target_x:.3f}m\n")

    state = "TRANSIT"
    transit_start_time = time.time()
    last_known_x = start_x

    try:
        while True:
            if state == "TRANSIT":
                # Drive straight
                send_commands(CRUISE_THRUST, 0.0)
                
                # Check ArUco Position
                x, y, yaw = locator.get_position()
                
                # Check if camera lost tags (position froze)
                if x == last_known_x:
                    status = "TAGS LOST - DRIVING BLIND"
                else:
                    status = "TAGS TRACKING"
                    last_known_x = x

                print(f"[TRANSIT] {status} | Target: {target_x:.2f}m | Current X: {x:.2f}m | Yaw: {math.degrees(yaw):.1f}°")
                
                # Check if we hit our -2 meter goal
                if x <= target_x:
                    print(f"\n>>> Reached Target X! Transitioning to ORBIT.")
                    state = "ORBIT"
                    
                # Safety timeout (in case tags are permanently lost and boat just drives into wall)
                if time.time() - transit_start_time > 15.0:
                    print(f"\n>>> Transit timeout reached! Forcing transition to ORBIT.")
                    state = "ORBIT"

            elif state == "ORBIT":
                # Hand over entirely to LiDAR, look specifically to the left
                angle_deg, dist_m = get_nearest_object_on_left()
                
                if dist_m is None:
                    print("[ORBIT] No buoy found on left side. Scanning...")
                    send_commands(CRUISE_THRUST, 0.0)
                else:
                    # Keep the buoy at -90 degrees (directly on the left)
                    dist_err = dist_m - TARGET_RADIUS
                    ang_err = angle_deg - (-90.0)
                    
                    rudder_cmd = np.clip((KP_DIST * dist_err) + (KP_ANG * ang_err), -45.0, 45.0)
                    send_commands(CRUISE_THRUST, rudder_cmd)
                    
                    print(f"[ORBIT] Locked on Buoy | Dist: {dist_m:.2f}m | Angle: {angle_deg:.1f}° | Steering: {rudder_cmd:.1f}°")

            time.sleep(0.1) # 10Hz debug loop

    except KeyboardInterrupt:
        print("\nManual Override Triggered.")
    finally:
        _running = False
        send_commands(0.0, 0.0)
        stop_all()
        locator.cleanup()
        print("Shutdown safe.")

if __name__ == "__main__":
    main()