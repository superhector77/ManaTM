#!/usr/bin/env python3
"""
main.py
-------
ManaTM Autonomous Boat — Hardware Deployment Node
Adapted from teammate's simulation script to use real sensors and actuators.

Run:
    source ~/servo_env/bin/activate
    python3 main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
import sys
import os
import time
import threading
import asyncio
import serial

# ==========================================
# LOCAL LIBRARY IMPORT SETUP
# ==========================================
LOCAL_BREEZYSLAM_DIR = './SLAM_Script'
if LOCAL_BREEZYSLAM_DIR not in sys.path:
    sys.path.insert(0, LOCAL_BREEZYSLAM_DIR)

try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import breezyslam locally.")
    exit(1)

# ── Sensor/actuator imports ────────────────────────────────────────────────────
import qwiic_ism330dhcx
import cv2
from gpiozero import Button as GPIOButton
from rplidarc1 import RPLidar
from Aruco import process_frame, get_robot_pose, T_cam1_in_robot, T_cam2_in_robot
from rudder_motor_pump import init as hat_init, set_rudder, set_motor, set_pump, stop_all
from thrustangle_to_motorservo_sig import thrust_angle_to_motor_servo
# from actuation_init import (
#     Pi5_LED, 
#     Lidar_Motor,
#     kill_switch_pressed
# )
from actuation_init import (
    Pi5Pixelbuf, set_light_state, get_light_state, run_lights,
    STATE_RAINBOW, STATE_LEAK, STATE_DONE,
    NEOPIXEL, NEOPIXEL_2, NUM_PIXELS, FLASH_INTERVAL
)
import board
import neopixel
from adafruit_led_animation.animation.rainbow import Rainbow
from adafruit_led_animation.animation.rainbowchase import RainbowChase
from adafruit_led_animation.animation.rainbowcomet import RainbowComet
from adafruit_led_animation.animation.rainbowsparkle import RainbowSparkle
from adafruit_led_animation.sequence import AnimationSequence


# ==========================================
# 1. PHYSICAL ENVIRONMENT & MISSION CONFIGURATION
# ==========================================
# --- RECTANGULAR TANK DIMENSIONS ---
TANK_WIDTH_X = 6.4   # Width of the physical tank in meters
TANK_LENGTH_Y = 2.67  # Length of the physical tank in meters

GRID_RES = 0.05

# Waypoints mapped to proportional rectangular coordinates
START_POS = (0, 0.5, 0) #-np.pi/2 # starting at 35 ft along the south wall, facing north
BUOY1_POS = (1.8 * TANK_WIDTH_X, 1.1 * TANK_LENGTH_Y) # at 41 ft, midway between north and south wall
BUOY2_POS = (3.6 * TANK_WIDTH_X, 1.1 * TANK_LENGTH_Y) # at 47 ft, midway between north and south wall

# # ArUco tags in the environment mapped to rectangular walls
# ARUCO_TAGS = {
#     1: (TANK_WIDTH_X, TANK_LENGTH_Y/2, np.pi),      # East wall center
#     2: (0, TANK_LENGTH_Y/2, 0),                     # West wall center
#     3: (TANK_WIDTH_X/2, TANK_LENGTH_Y, -np.pi/2),   # North wall center
#     4: (TANK_WIDTH_X/2, 0, np.pi/2)                 # South wall center
# }

# ==========================================
# 2. HARDWARE CONSTRAINTS & SENSOR SPECS
# ==========================================
MAX_SPEED = 0.5                   
MAX_THRUST_LIMIT = 30.0           
MAX_STEER = np.radians(45)        

BOAT_LENGTH = 0.762       
BOAT_WIDTH = 0.2
REAR_AXIS_OFFSET = BOAT_LENGTH / 5.0   

LIDAR_RAYS = 1080                 
LIDAR_MAX_RANGE = 5.0             

# class PhysicalLidar(Laser):
#     def __init__(self):
#         super().__init__(LIDAR_RAYS, 10, 360, 0) 

# ==========================================
# 3. NAVIGATION & CONTROL TUNING PARAMETERS
# ==========================================
KP_THRUST = 15.0
KI_THRUST = 1.0

KP_STEER = 3.0     
KI_STEER = 0.1    
KD_STEER = 1.5   

VFH_SAFE_DIST = 3.0       
VFH_CRITICAL_DIST = 1.5   
VFH_BRAKE_SPEED = 0.2     

ARUCO_FUSION_WEIGHT = 0.5 

# ==========================================
# HARDWARE CONFIG
# ==========================================
IMU_ADDRESS     = 0x6B
LEAK_SENSOR_PIN = 4
LIDAR_PORT      = '/dev/ttyUSB0'
LIDAR_BAUD      = 460800
CAM1_INDEX      = 0
CAM2_INDEX      = 1
CAM_WIDTH       = 640
CAM_HEIGHT      = 480
CAM_FPS         = 30

# ==========================================
# SHARED SENSOR BUFFERS
# ==========================================
_lidar_buffer  = []
_lidar_lock    = threading.Lock()

_imu_buffer    = {"accel": (0.0, 0.0, 0.0), "gyro": (0.0, 0.0, 0.0)}
_imu_lock      = threading.Lock()

_aruco_buffer  = None
_aruco_lock    = threading.Lock()

_leak_detected = False
_leak_lock     = threading.Lock()

_running       = True


# ==========================================
# HARDWARE INITIALISATION
# ==========================================
def init_imu():
    print("Initialising IMU...")
    imu = qwiic_ism330dhcx.QwiicISM330DHCX(address=IMU_ADDRESS)
    imu.begin()
    imu.device_reset()
    while imu.get_device_reset() == False:
        time.sleep(1)
    time.sleep(0.1)
    imu.set_device_config()
    imu.set_block_data_update()
    imu.set_accel_data_rate(imu.kXlOdr104Hz)
    imu.set_accel_full_scale(imu.kXlFs4g)
    imu.set_accel_filter_lp2()
    imu.set_accel_slope_filter(imu.kLpOdrDiv100)
    imu.set_gyro_data_rate(imu.kGyroOdr104Hz)
    imu.set_gyro_full_scale(imu.kGyroFs500dps)
    imu.set_gyro_filter_lp1()
    imu.set_gyro_lp1_bandwidth(imu.kBwMedium)
    print("IMU ready")
    return imu


def init_cameras():
    print("Initialising cameras...")
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    if not cap1.isOpened():
        print("Camera 1 not found", file=sys.stderr)
        sys.exit(1)
    if not cap2.isOpened():
        print("Camera 2 not found — continuing with single camera")
    print("Cameras ready")
    return cap1, cap2


def init_lidar():
    print("Initialising LiDAR...")
    try:
        ser = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.close()
        time.sleep(0.5)
    except Exception as e:
        print(f"[LIDAR] pre-flush warning: {e}")
    lidar = RPLidar(port=LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=0.2)
    print("LiDAR ready")
    return lidar


def init_pixels():
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


# ==========================================
# BACKGROUND SENSOR THREADS
# ==========================================
def imu_thread(imu):
    global _running
    print("[IMU] thread started")
    while _running:
        if imu.check_status():
            accel = imu.get_accel()
            gyro  = imu.get_gyro()
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
    global _running, _aruco_buffer
    print("[CAM] thread started")
    while _running:
        pose = None
        ret1, frame1 = cap1.read()
        if ret1:
            pos1, rot1 = process_frame(frame1, "Cam1")
            if pos1 is not None:
                robot_pos, robot_rot = get_robot_pose(pos1, rot1, T_cam1_in_robot)
                yaw = np.arctan2(robot_rot[1, 0], robot_rot[0, 0])
                pose = (float(robot_pos[0]), float(robot_pos[1]),
                        float(robot_pos[2]), float(yaw))
        if cap2.isOpened() and pose is None:
            ret2, frame2 = cap2.read()
            if ret2:
                pos2, rot2 = process_frame(frame2, "Cam2")
                if pos2 is not None:
                    robot_pos, robot_rot = get_robot_pose(pos2, rot2, T_cam2_in_robot)
                    yaw = np.arctan2(robot_rot[1, 0], robot_rot[0, 0])
                    pose = (float(robot_pos[0]), float(robot_pos[1]),
                            float(robot_pos[2]), float(yaw))
        with _aruco_lock:
            _aruco_buffer = pose
        time.sleep(0.033)
    print("[CAM] thread stopped")


def leak_thread(leak_sensor):
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


async def _lidar_queue_reader(lidar):
    global _running, _lidar_buffer
    while _running:
        if lidar.output_queue.qsize() < 1:
            await asyncio.sleep(0.05)
            continue
        point = await lidar.output_queue.get()
        if point.get('q', 0) > 0:
            angle_deg = point['a_deg']
            dist_m    = point['d_mm'] / 1000.0
            with _lidar_lock:
                _lidar_buffer.append([angle_deg, dist_m])
                if len(_lidar_buffer) > 1080:
                    _lidar_buffer = _lidar_buffer[-1080:]


async def _lidar_worker(lidar):
    global _running
    print("[LIDAR] worker started")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(lidar.simple_scan(make_return_dict=True))
            tg.create_task(_lidar_queue_reader(lidar))
    except Exception as e:
        print(f"[LIDAR] error: {e}")
    print("[LIDAR] worker stopped")


def lidar_thread(lidar):
    asyncio.run(_lidar_worker(lidar))


def lights_thread(pixels, animations):
    print("[LIGHTS] thread started")
    run_lights(pixels, animations)


# ==========================================
# REAL SENSOR READ FUNCTIONS
# (replace mock functions from original)
# ==========================================
def get_real_lidar_data():
    with _lidar_lock:
        if not _lidar_buffer:
            return [[i * (360.0 / 1080), None] for i in range(1080)]
        return list(_lidar_buffer)


def get_real_imu_data():
    with _imu_lock:
        return _imu_buffer["accel"], _imu_buffer["gyro"]


def get_real_aruco_data():
    with _aruco_lock:
        return _aruco_buffer


def get_leak():
    with _leak_lock:
        return _leak_detected


# ==========================================
# COMMAND TRANSLATION
# ==========================================
def send_motor_commands(thrust_cmd, prop_angle_cmd):
    if get_leak():
        set_motor(0.0)
        set_rudder(90)
        return
    motor_sig, servo_sig = thrust_angle_to_motor_servo(thrust_cmd, prop_angle_cmd)
    motor_sig = float(np.clip(motor_sig, 0.0, 1.0))
    servo_sig = float(np.clip(servo_sig, 0.0, 180.0))
    set_motor(motor_sig)
    set_rudder(servo_sig)


# ==========================================
# AUTONOMOUS NAVIGATOR 
# ==========================================
class PhysicalLidar(Laser):
    def __init__(self):
        super().__init__(LIDAR_RAYS, 10, 360, 0)



    # ── paste your teammate's AutonomousNavigator class here unchanged ──

# ==========================================
# AUTONOMOUS NAVIGATOR (The Brain)
# ==========================================
class AutonomousNavigator:
    def __init__(self, start_pos=START_POS):
        self.laser_spec = PhysicalLidar()
        
        self.MAP_SIZE_SLAM = 30.0
        self.GRID_DIMS_SLAM = int(self.MAP_SIZE_SLAM / GRID_RES)
        
        self.slam = RMHC_SLAM(self.laser_spec, self.GRID_DIMS_SLAM, int(self.MAP_SIZE_SLAM), 
                              sigma_xy_mm=250, sigma_theta_degrees=1, max_search_iter=2000, random_seed=42)
        
        self.slam_start_x = self.MAP_SIZE_SLAM / 2.0
        self.slam_start_y = self.MAP_SIZE_SLAM / 2.0
        
        self.slam.position.x_mm = self.slam_start_x * 1000.0
        self.slam.position.y_mm = self.slam_start_y * 1000.0
        self.slam.position.theta_degrees = np.degrees(start_pos[2])
        
        self.slam_offset_x = start_pos[0] - self.slam_start_x
        self.slam_offset_y = start_pos[1] - self.slam_start_y
        
        self.est_x, self.est_y, self.est_theta = start_pos 
        
        self.last_time = time.time()
        self.u = 0.0 
        
        self.lidar_ranges = np.zeros(LIDAR_RAYS)
        self.lidar_angles = np.zeros(LIDAR_RAYS)
        self.slam_map = np.zeros((self.GRID_DIMS_SLAM, self.GRID_DIMS_SLAM))
        
        self.integral_heading_error = 0.0
        self.integral_speed_error = 0.0
        
        self.current_thrust = 0.0
        self.current_angle = 0.0
        
        # Mapped to new rectangular dimensions
        self.start_dock = (start_pos[0], start_pos[1])
        self.end_dock = (0.9 * TANK_WIDTH_X, 0.1 * TANK_LENGTH_Y)
        self.expected_buoy1 = BUOY1_POS
        self.expected_buoy2 = BUOY2_POS
        self.rtb_target = (0.2 * TANK_WIDTH_X, 0.8 * TANK_LENGTH_Y)
        
        clearance_x = self.start_dock[0] + 0.5 * np.cos(start_pos[2])
        clearance_y = self.start_dock[1] + 0.5 * np.sin(start_pos[2])
        self.path_leave_dock1 = [(clearance_x, clearance_y)]
        self.path_approach_dock1 = [self.rtb_target, self.start_dock]
        self.path_approach_dock2 = [(self.end_dock[0], self.end_dock[1] + 0.1 * TANK_LENGTH_Y), self.end_dock]
        
        self.state = "START" 
        self.wp_index = 0
        
        self.search_angle = 0.0 
        self.tracked_buoy1 = None  
        self.tracked_buoy2 = None
        self.total_circled_angle = 0.0
        self.last_buoy_angle = 0.0
        
        self.fig8_target_buoy = 1
        self.fig8_direction = -1 
        self.fig8_crossings = 0
        self.fig8_ready_to_swap = False
        
        self.history_dict = {
            "START": ([], []), 
            "LEAVE_DOCK1": ([], []), "WANDER": ([], []), "SEARCH_BUOY1": ([], []),
            "CIRCLE_BUOY1": ([], []), "APPROACH_DOCK1": ([], []), "CIRCLE_DOCK1": ([], []),
            "FIND_BUOYS": ([], []), "FIGURE_8": ([], []), "APPROACH_DOCK2": ([], []),
            "CIRCLE_DOCK2": ([], []), "RETURN_TO_BASE": ([], []), "E_STOP": ([], [])
        }

    def get_buoy_update(self, tracked_pos, expected_pos):
        ref_pos = tracked_pos if tracked_pos else expected_pos
        best_idx = -1
        min_err = float('inf')
        
        for i, r in enumerate(self.lidar_ranges):
            if r < LIDAR_MAX_RANGE:
                ang = self.est_theta + self.lidar_angles[i]
                lx = self.est_x + r*np.cos(ang)
                ly = self.est_y + r*np.sin(ang)
                err = np.hypot(lx - ref_pos[0], ly - ref_pos[1])
                if err < min_err:
                    min_err = err
                    best_idx = i

        if best_idx != -1 and min_err < 2.0: 
            r = self.lidar_ranges[best_idx]
            ang = self.est_theta + self.lidar_angles[best_idx]
            new_pos = (self.est_x + r*np.cos(ang), self.est_y + r*np.sin(ang))
            return new_pos, r, self.lidar_angles[best_idx]
        else:
            dist = np.hypot(ref_pos[0] - self.est_x, ref_pos[1] - self.est_y)
            ang = (np.arctan2(ref_pos[1] - self.est_y, ref_pos[0] - self.est_x) - self.est_theta)
            ang = (ang + np.pi) % (2*np.pi) - np.pi
            return ref_pos, dist, ang

    def step(self, lidar_matrix, accel_tuple, gyro_tuple, aruco_data):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0: dt = 0.001 
        self.last_time = current_time

        ax, ay, az = accel_tuple
        gx, gy, gz = gyro_tuple

        # --- 1. SENSOR FUSION & SLAM ESTIMATION ---
        scan_mm = []
        parsed_angles = []
        parsed_ranges = []
        
        for row in lidar_matrix:
            angle_deg, dist_m = row[0], row[1]
            parsed_angles.append(np.radians(angle_deg))
            if dist_m is None:
                scan_mm.append(0) 
                parsed_ranges.append(LIDAR_MAX_RANGE)
            else:
                scan_mm.append(int(dist_m * 1000.0))
                parsed_ranges.append(dist_m)
                
        self.lidar_angles = np.array(parsed_angles)
        self.lidar_ranges = np.array(parsed_ranges)

        dxy_mm = self.u * 1000.0 * dt
        dtheta_degrees = np.degrees(gz) * dt

        self.slam.update(scan_mm, pose_change=(dxy_mm, dtheta_degrees, dt))

        x_mm, y_mm, theta_deg = self.slam.getpos()
        
        last_x, last_y = self.est_x, self.est_y
        self.est_x = (x_mm / 1000.0) + self.slam_offset_x
        self.est_y = (y_mm / 1000.0) + self.slam_offset_y
        self.est_theta = np.radians(theta_deg)

        # --- ARUCO SENSOR FUSION (COMPLEMENTARY FILTER) ---
        if aruco_data is not None:
            a_x, a_y, a_z, a_yaw = aruco_data
            alpha = ARUCO_FUSION_WEIGHT  
            
            self.est_x = (1 - alpha) * self.est_x + alpha * a_x
            self.est_y = (1 - alpha) * self.est_y + alpha * a_y
            
            curr_sin, curr_cos = np.sin(self.est_theta), np.cos(self.est_theta)
            a_sin, a_cos = np.sin(a_yaw), np.cos(a_yaw)
            self.est_theta = np.arctan2((1 - alpha)*curr_sin + alpha*a_sin, 
                                        (1 - alpha)*curr_cos + alpha*a_cos)
            
            # Sync internal SLAM state to reality
            self.slam.position.x_mm = (self.est_x - self.slam_offset_x) * 1000.0
            self.slam.position.y_mm = (self.est_y - self.slam_offset_y) * 1000.0
            self.slam.position.theta_degrees = np.degrees(self.est_theta)

        dx = self.est_x - last_x
        dy = self.est_y - last_y
        inst_u = (dx * np.cos(self.est_theta) + dy * np.sin(self.est_theta)) / dt
        self.u = (0.8 * self.u) + (0.2 * inst_u)

        mapbytes = bytearray(self.GRID_DIMS_SLAM * self.GRID_DIMS_SLAM)
        self.slam.getmap(mapbytes)
        self.slam_map = np.array(mapbytes).reshape((self.GRID_DIMS_SLAM, self.GRID_DIMS_SLAM))
        
        # --- 2. COGNITIVE MISSION STATE MACHINE ---
        if self.state in ["START", "MISSION_COMPLETE", "E_STOP"]:
            self.current_thrust = 0.0
            self.current_angle = 0.0
            self.integral_heading_error = 0.0
            self.integral_speed_error = 0.0
            return 0.0, 0.0
            
        target_v = 0.0
        target_heading = self.est_theta

        if self.state == "LEAVE_DOCK1":
            target = self.path_leave_dock1[self.wp_index]
            dist = np.hypot(target[0] - self.est_x, target[1] - self.est_y)
            if dist < 0.2:
                if self.wp_index == len(self.path_leave_dock1) - 1: 
                    self.state = "WANDER"
                else:
                    self.wp_index += 1
            pp_heading = np.arctan2(target[1] - self.est_y, target[0] - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED)

        elif self.state == "WANDER":
            self.wander_angle += np.random.normal(0, 0.1)
            self.wander_angle = (self.wander_angle + np.pi) % (2 * np.pi) - np.pi
            target_v, target_heading = self._arbiter(self.wander_angle, MAX_SPEED * 0.7)

        elif self.state == "RETURN_TO_BASE":
            dx_p = self.rtb_target[0] - self.est_x
            dy_p = self.rtb_target[1] - self.est_y
            dist = np.hypot(dx_p, dy_p)
            angle_to_rtb = np.arctan2(dy_p, dx_p)
            if dist > 1.5:
                target_v, target_heading = self._arbiter(angle_to_rtb, MAX_SPEED)
            else:
                target_radius = 1.0
                direction = -1 
                K = 1.5
                alpha_ang = np.arctan(K * (dist - target_radius))
                tangent_offset = direction * (np.pi/2 - alpha_ang)
                orbital_heading = angle_to_rtb + tangent_offset
                target_v, target_heading = self._arbiter(orbital_heading, MAX_SPEED * 0.5)

        elif self.state == "SEARCH_BUOY1":
            self.search_angle += 0.05
            search_radius = 0.5 + (self.search_angle * 0.05)
            sx = self.expected_buoy1[0] + search_radius * np.cos(self.search_angle)
            sy = self.expected_buoy1[1] + search_radius * np.sin(self.search_angle)
            
            pp_heading = np.arctan2(sy - self.est_y, sx - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED * 0.7)

            differences = np.abs(np.diff(self.lidar_ranges))
            valid_mask = (self.lidar_ranges[:-1] < LIDAR_MAX_RANGE - 1.0) & (self.lidar_ranges[1:] < LIDAR_MAX_RANGE - 1.0)
            if np.any((differences > 1.0) & (self.lidar_ranges[:-1] < 3) & valid_mask):
                self.state = "CIRCLE_BUOY1"
                self.tracked_buoy1, _, _ = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
                self.total_circled_angle = 0.0
                self.last_buoy_angle = np.arctan2(self.est_y - self.tracked_buoy1[1], self.est_x - self.tracked_buoy1[0])

        elif self.state == "CIRCLE_BUOY1":
            self.tracked_buoy1, _, _ = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
            target_radius = 1.5  
            direction = -1 
            
            dx_p = self.tracked_buoy1[0] - self.est_x
            dy_p = self.tracked_buoy1[1] - self.est_y
            d = np.hypot(dx_p, dy_p)
            angle_to_buoy = np.arctan2(dy_p, dx_p)
            
            K = 1.5
            alpha_ang = np.arctan(K * (d - target_radius))
            tangent_offset = direction * (np.pi/2 - alpha_ang)
            orbital_heading = angle_to_buoy + tangent_offset
            
            target_v, target_heading = self._arbiter(orbital_heading, MAX_SPEED, ignore_target=self.tracked_buoy1)
            
            current_buoy_angle = np.arctan2(self.est_y - self.tracked_buoy1[1], self.est_x - self.tracked_buoy1[0])
            angle_diff = (current_buoy_angle - self.last_buoy_angle + np.pi) % (2*np.pi) - np.pi
            self.total_circled_angle += angle_diff
            self.last_buoy_angle = current_buoy_angle
            
            if abs(self.total_circled_angle) >= (3.0 * 2.0 * np.pi):
                self.state = "APPROACH_DOCK1"
                self.wp_index = 0

        elif self.state == "APPROACH_DOCK1":
            target = self.path_approach_dock1[self.wp_index]
            dist = np.hypot(target[0] - self.est_x, target[1] - self.est_y)
            if dist < 1.0:
                if self.wp_index == len(self.path_approach_dock1) - 1:
                    self.state = "CIRCLE_DOCK1"
                else:
                    self.wp_index += 1
            pp_heading = np.arctan2(target[1] - self.est_y, target[0] - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED)

        elif self.state in ["CIRCLE_DOCK1", "CIRCLE_DOCK2"]:
            target = self.start_dock if self.state == "CIRCLE_DOCK1" else self.end_dock
            dx_p = target[0] - self.est_x
            dy_p = target[1] - self.est_y
            d = np.hypot(dx_p, dy_p)
            angle_to_dock = np.arctan2(dy_p, dx_p)
            
            target_radius = 1.5  
            direction = -1 
            K = 1.5
            alpha_ang = np.arctan(K * (d - target_radius))
            tangent_offset = direction * (np.pi/2 - alpha_ang)
            orbital_heading = angle_to_dock + tangent_offset
            
            target_v, target_heading = self._arbiter(orbital_heading, MAX_SPEED)

        elif self.state == "FIND_BUOYS":
            mid_x = (self.expected_buoy1[0] + self.expected_buoy2[0]) / 2.0
            mid_y = (self.expected_buoy1[1] + self.expected_buoy2[1]) / 2.0
            dist = np.hypot(mid_x - self.est_x, mid_y - self.est_y)
            
            self.tracked_buoy1, _, _ = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
            self.tracked_buoy2, _, _ = self.get_buoy_update(self.tracked_buoy2, self.expected_buoy2)
            
            if dist < 4.0:
                self.state = "FIGURE_8"
                self.fig8_target_buoy = 1
                self.fig8_direction = -1 
                self.fig8_crossings = 0
                self.fig8_ready_to_swap = False
            
            pp_heading = np.arctan2(mid_y - self.est_y, mid_x - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED)

        elif self.state == "FIGURE_8":
            tb1, d1, a1 = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
            self.tracked_buoy1 = tb1
            tb2, d2, a2 = self.get_buoy_update(self.tracked_buoy2, self.expected_buoy2)
            self.tracked_buoy2 = tb2
            
            D_buoys = np.hypot(self.tracked_buoy1[0] - self.tracked_buoy2[0], 
                               self.tracked_buoy1[1] - self.tracked_buoy2[1])
            
            d_inactive = d2 if self.fig8_target_buoy == 1 else d1
            if d_inactive > (D_buoys + 0.5):
                self.fig8_ready_to_swap = True
                
            swap_threshold = D_buoys - 2.0 + 0.8 
            if self.fig8_ready_to_swap and d_inactive < swap_threshold:
                self.fig8_ready_to_swap = False
                self.fig8_crossings += 1
                self.fig8_target_buoy = 2 if self.fig8_target_buoy == 1 else 1
                self.fig8_direction *= -1
                
            if self.fig8_crossings >= 6: 
                self.state = "APPROACH_DOCK2"
                self.wp_index = 0
            else:
                active_target = tb1 if self.fig8_target_buoy == 1 else tb2
                target_radius = 1.5  
                direction = self.fig8_direction
                
                dx_p = active_target[0] - self.est_x
                dy_p = active_target[1] - self.est_y
                d = np.hypot(dx_p, dy_p)
                angle_to_buoy = np.arctan2(dy_p, dx_p)
                
                K = 1.5
                alpha_ang = np.arctan(K * (d - target_radius))
                tangent_offset = direction * (np.pi/2 - alpha_ang)
                orbital_heading = angle_to_buoy + tangent_offset
                
                target_v, target_heading = self._arbiter(orbital_heading, MAX_SPEED, ignore_target=active_target)

        elif self.state == "APPROACH_DOCK2":
            target = self.path_approach_dock2[self.wp_index] 
            dist = np.hypot(target[0] - self.est_x, target[1] - self.est_y)
            if dist < 1.0:
                if self.wp_index == len(self.path_approach_dock2) - 1:
                    self.state = "CIRCLE_DOCK2"
                else:
                    self.wp_index += 1
            pp_heading = np.arctan2(target[1] - self.est_y, target[0] - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED)

        # --- LOW-LEVEL ROBUST CONTROLLER (PID) ---
        heading_error = target_heading - self.est_theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        speed_error = target_v - self.u
        
        self.integral_heading_error += heading_error * dt
        self.integral_heading_error = np.clip(self.integral_heading_error, -10.0, 10.0) 
        
        self.integral_speed_error += speed_error * dt
        self.integral_speed_error = np.clip(self.integral_speed_error, -20.0, 20.0)
        
        prop_angle_cmd = np.clip(
            (KP_STEER * heading_error) + 
            (KI_STEER * self.integral_heading_error) - 
            (KD_STEER * gz), 
            -MAX_STEER, MAX_STEER
        )

        base_thrust = (KP_THRUST * speed_error) + (KI_THRUST * self.integral_speed_error)
        base_thrust = max(base_thrust, 0.0)
        
        turn_assist_thrust = (abs(prop_angle_cmd) / MAX_STEER) * (15.0 * target_v) 
        thrust_cmd = np.clip(max(base_thrust, turn_assist_thrust), 0.0, MAX_THRUST_LIMIT)

        self.current_thrust = thrust_cmd
        self.current_angle = prop_angle_cmd

        if self.state in self.history_dict:
            self.history_dict[self.state][0].append(self.est_x)
            self.history_dict[self.state][1].append(self.est_y)

        return thrust_cmd, prop_angle_cmd

    def _arbiter(self, desired_heading, max_speed, ignore_target=None):
        hist_bins = 72 
        histogram = np.zeros(hist_bins)
        
        valid_ranges = np.copy(self.lidar_ranges)
        if ignore_target is not None:
            for i, r in enumerate(self.lidar_ranges):
                if r < VFH_SAFE_DIST:
                    ray_ang = self.est_theta + self.lidar_angles[i]
                    rx = self.est_x + r * np.cos(ray_ang)
                    ry = self.est_y + r * np.sin(ray_ang)
                    
                    if np.hypot(rx - ignore_target[0], ry - ignore_target[1]) < 1.0:
                        valid_ranges[i] = LIDAR_MAX_RANGE
        
        min_dist = np.min(valid_ranges)
        MIN_SPEED = 0.05 
        
        if min_dist < VFH_CRITICAL_DIST:
            left_openness = np.mean([r for i, r in enumerate(valid_ranges) if self.lidar_angles[i] > 0])
            right_openness = np.mean([r for i, r in enumerate(valid_ranges) if self.lidar_angles[i] < 0])
            escape_dir = 1.0 if left_openness > right_openness else -1.0
            return MIN_SPEED, self.est_theta + (escape_dir * np.pi/2)

        for i, r in enumerate(valid_ranges):
            if r < VFH_SAFE_DIST:
                global_angle = (self.est_theta + self.lidar_angles[i]) % (2*np.pi)
                bin_idx = int((global_angle / (2*np.pi)) * hist_bins)
                block_width = int(4 + (VFH_SAFE_DIST - r) * 5) 
                for b in range(-block_width, block_width + 1):
                    histogram[(bin_idx + b) % hist_bins] = 1

        candidates = [i * (2*np.pi / hist_bins) for i in range(hist_bins) if histogram[i] == 0]
        if not candidates:
            return MIN_SPEED, self.est_theta + np.pi/2
            
        desired_wrapped = desired_heading % (2*np.pi)
        def ang_dist(a, b):
            diff = abs(a - b)
            return min(diff, 2*np.pi - diff)
            
        best_heading = min(candidates, key=lambda h: ang_dist(h, desired_wrapped))
        
        path_clearance = 5.0 
        for i, r in enumerate(valid_ranges):
            ray_global_angle = (self.est_theta + self.lidar_angles[i]) % (2*np.pi)
            if ang_dist(ray_global_angle, best_heading) < np.pi/6: 
                if r < path_clearance:
                    path_clearance = r
                    
        LOOKAHEAD = VFH_SAFE_DIST + 1.0
        CRITICAL = VFH_CRITICAL_DIST
        
        if path_clearance >= LOOKAHEAD:
            path_speed_factor = 1.0 
        elif path_clearance <= CRITICAL:
            path_speed_factor = VFH_BRAKE_SPEED 
        else:
            ratio = (path_clearance - CRITICAL) / (LOOKAHEAD - CRITICAL)
            path_speed_factor = VFH_BRAKE_SPEED + (1.0 - VFH_BRAKE_SPEED) * ratio
            
        current_wrapped = self.est_theta % (2*np.pi)
        turn_severity = ang_dist(best_heading, current_wrapped)
        turn_speed_factor = max(0.1, 1.0 - (turn_severity / (np.pi / 4.0)))
        
        final_speed = max_speed * path_speed_factor * turn_speed_factor
        return max(MIN_SPEED, final_speed), best_heading


# ==========================================
# MAIN
# ==========================================
def main():
    global _running

    print("=" * 60)
    print("  ManaTM Autonomous Boat  —  Startup")
    print("=" * 60)

    # Initialise hardware
    imu          = init_imu()
    cap1, cap2   = init_cameras()
    leak_sensor  = GPIOButton(LEAK_SENSOR_PIN, pull_up=False)
    lidar        = init_lidar()
    pixels, animations = init_pixels()

    print("Initialising servo pHAT...")
    hat_init()
    print("Servo pHAT ready\n")

    # Start background threads
    threads = [
        threading.Thread(target=imu_thread,    args=(imu,),              daemon=True),
        threading.Thread(target=camera_thread, args=(cap1, cap2),        daemon=True),
        threading.Thread(target=leak_thread,   args=(leak_sensor,),      daemon=True),
        threading.Thread(target=lidar_thread,  args=(lidar,),            daemon=True),
        threading.Thread(target=lights_thread, args=(pixels, animations), daemon=True),
    ]
    for t in threads:
        t.start()

    print("Warming up sensors (3s)...")
    time.sleep(3.0)

    navigator = AutonomousNavigator()
    set_light_state(STATE_RAINBOW)
    print("Navigator ready. Use the UI to start missions.\n")

    # ==========================================
    # VISUALIZATION LOOP (matplotlib on main thread)
    # ==========================================
    plt.ion()
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='polar')
    plt.subplots_adjust(bottom=0.15)

    # Buttons
    ax_start = plt.axes([0.16, 0.02, 0.15, 0.06])
    btn_start = Button(ax_start, 'START M1: CIRCLE', color='forestgreen', hovercolor='limegreen')
    btn_start.label.set_color('white')
    btn_start.label.set_fontweight('bold')

    ax_pause = plt.axes([0.33, 0.02, 0.08, 0.06])
    btn_pause = Button(ax_pause, 'PAUSE NAV', color='gold', hovercolor='yellow')
    btn_pause.label.set_fontweight('bold')

    ax_home = plt.axes([0.43, 0.02, 0.11, 0.06])
    btn_home = Button(ax_home, 'RETURN HOME', color='darkorange', hovercolor='orange')
    btn_home.label.set_color('white')
    btn_home.label.set_fontweight('bold')

    ax_estop = plt.axes([0.56, 0.02, 0.08, 0.06])
    btn_estop = Button(ax_estop, 'E-STOP', color='red', hovercolor='darkred')
    btn_estop.label.set_color('white')
    btn_estop.label.set_fontweight('bold')

    mission_stage = [0]
    def start_callback(event):
        if mission_stage[0] == 0:
            navigator.state = "LEAVE_DOCK1"
            navigator.wp_index = 0
            btn_start.label.set_text('START M1: CIRCLE')
            mission_stage[0] = 1
            print("\n[>] MISSION STARTED: Leaving Dock 1.")
        elif mission_stage[0] == 1:
            navigator.state = "SEARCH_BUOY1"
            navigator.wp_index = 0
            btn_start.label.set_text('START M2: FIG-8')
            mission_stage[0] = 2
            print("\n[>] MISSION 1 STARTED: Circling Buoy 1.")
        elif mission_stage[0] == 2:
            navigator.state = "FIND_BUOYS"
            navigator.wp_index = 0
            btn_start.label.set_text('MISSIONS ACTIVE')
            btn_start.color = 'gray'
            mission_stage[0] = 3
            print("\n[>] MISSION 2 STARTED: Figure-Eight.")
    btn_start.on_clicked(start_callback)

    pause_state = {"paused": False}
    def pause_callback(event):
        pause_state["paused"] = not pause_state["paused"]
        btn_pause.label.set_text("RESUME NAV" if pause_state["paused"] else "PAUSE NAV")
        if pause_state["paused"]:
            set_motor(0.0)
            set_rudder(90)
    btn_pause.on_clicked(pause_callback)

    def home_callback(event):
        navigator.state = "RETURN_TO_BASE"
        print("\n[<] RETURN HOME TRIGGERED.")
    btn_home.on_clicked(home_callback)

    def estop_callback(event):
        navigator.state = "E_STOP"
        set_motor(0.0)
        set_rudder(90)
        set_light_state(STATE_LEAK)
        print("\n[!] E-STOP TRIGGERED.")
    btn_estop.on_clicked(estop_callback)

    # Throttle and rudder bars
    ax_throttle = plt.axes([0.08, 0.03, 0.06, 0.04])
    ax_throttle.set_xlim(0, 50)
    ax_throttle.set_yticks([])
    ax_throttle.set_title("Thrust Cmd (N)", fontsize=10)
    throttle_bar = ax_throttle.barh([0], [0], color='limegreen')

    ax_rudder = plt.axes([0.66, 0.03, 0.12, 0.04])
    ax_rudder.set_xlim(-0.8, 0.8)
    ax_rudder.axvline(0, color='black', linewidth=1)
    ax_rudder.set_yticks([])
    ax_rudder.set_title("Prop Angle Cmd (Rad)", fontsize=10)
    rudder_bar = ax_rudder.barh([0], [0], color='dodgerblue')

    # # Plot setup
    # ax1.set_title("Physical Tank - Global Estimate")
    # ax1.set_xlim(0, MAP_SIZE); ax1.set_ylim(0, MAP_SIZE)
    # ax1.grid(True, linestyle='--', alpha=0.5)
    # ax1.plot(navigator.rtb_target[0],  navigator.rtb_target[1],  'o',  color='orange', markersize=4,  label="RTB")
    # ax1.plot(navigator.end_dock[0],    navigator.end_dock[1],    'rs', markersize=12, label="Dock 2")
    # ax1.plot(navigator.start_dock[0],  navigator.start_dock[1],  'gs', markersize=12, label="Dock 1")

    # boat_rect1 = patches.Rectangle(
    #     (-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH,
    #     linewidth=1, edgecolor='black', facecolor='orange', alpha=0.8)
    # ax1.add_patch(boat_rect1)
    # lidar_lines, = ax1.plot([], [], color='red', alpha=0.3, linewidth=0.5)

    # ax2.set_title("BreezySLAM Map")
    # ax2.set_xlim(0, MAP_SIZE); ax2.set_ylim(0, MAP_SIZE)
    # slam_img = ax2.imshow(
    #     np.zeros((navigator.GRID_DIMS_SLAM, navigator.GRID_DIMS_SLAM)),
    #     origin='lower', cmap='Blues',
    #     extent=[navigator.slam_offset_x,
    #             navigator.MAP_SIZE_SLAM + navigator.slam_offset_x,
    #             navigator.slam_offset_y,
    #             navigator.MAP_SIZE_SLAM + navigator.slam_offset_y],
    #     vmin=0, vmax=255)

    # maneuver_colors = {
    #     "LEAVE_DOCK1": "green",   "WANDER": "gray",       "SEARCH_BUOY1": "purple",
    #     "CIRCLE_BUOY1": "cyan",   "APPROACH_DOCK1": "green", "CIRCLE_DOCK1": "green",
    #     "FIND_BUOYS": "purple",   "FIGURE_8": "magenta",  "APPROACH_DOCK2": "red",
    #     "CIRCLE_DOCK2": "red",    "RETURN_TO_BASE": "orange", "E_STOP": "black"
    # }
    # history_lines = {}
    # for state, color in maneuver_colors.items():
    #     line, = ax2.plot([], [], color=color, marker='.', linestyle='None',
    #                      markersize=3, alpha=0.6, zorder=2)
    #     history_lines[state] = line

    # boat_rect2 = patches.Rectangle(
    #     (-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH,
    #     linewidth=1, edgecolor='black', facecolor='orange', alpha=1.0)
    # ax2.add_patch(boat_rect2)
    # ax2.plot(navigator.expected_buoy1[0], navigator.expected_buoy1[1], 'yx', markersize=10)
    # ax2.plot(navigator.expected_buoy2[0], navigator.expected_buoy2[1], 'yx', markersize=10)
    # buoy1_pt, = ax2.plot([], [], 'go', markersize=8)
    # buoy2_pt, = ax2.plot([], [], 'mo', markersize=8)

    ax3.set_title("LiDAR (Local)")
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(1)
    ax3.set_ylim(0, LIDAR_MAX_RANGE)
    ax3.set_yticks(np.arange(1, LIDAR_MAX_RANGE + 1, 1.0))
    ax3.tick_params(axis='y', labelsize=8)
    lidar_polar = ax3.scatter(np.zeros(LIDAR_RAYS), np.zeros(LIDAR_RAYS),
                              c='red', s=15, alpha=0.8)
    ax3.plot(0, 0, marker='^', color='blue', markersize=12)

    # ── Main loop ──────────────────────────────────────────────────────────────
    step = 0
    try:
        while True:
            if pause_state["paused"]:
                plt.pause(0.05)
                continue

            # 1. Read real sensors
            lidar_data          = get_real_lidar_data()
            imu_accel, imu_gyro = get_real_imu_data()
            aruco_data          = get_real_aruco_data()

            # 2. Safety check
            if get_leak():
                set_motor(0.0)
                set_rudder(90)
                plt.pause(0.05)
                continue

            # 3. Run navigator
            cmd_thrust, cmd_angle = navigator.step(
                lidar_data, imu_accel, imu_gyro, aruco_data
            )

            # 4. Send to hardware
            send_motor_commands(cmd_thrust, cmd_angle)

            # 5. Sync light state
            if navigator.state == "MISSION_COMPLETE":
                set_light_state(STATE_DONE)
            elif navigator.state != "E_STOP" and not get_leak():
                set_light_state(STATE_RAINBOW)

            # 6. Update UI every other step
            if step % 2 == 0:
                throttle_bar[0].set_width(navigator.current_thrust)
                throttle_bar[0].set_color(
                    'red' if navigator.current_thrust > 25 else 'limegreen')
                rudder_bar[0].set_width(navigator.current_angle)

                state_disp = navigator.state
                if navigator.state == "CIRCLE_BUOY1":
                    loops = int(abs(navigator.total_circled_angle) / (2*np.pi))
                    state_disp += f" ({loops}/3 loops)"
                elif navigator.state in ["CIRCLE_DOCK1", "CIRCLE_DOCK2"]:
                    state_disp += " (Holding position)"
                elif navigator.state == "FIGURE_8":
                    cycles = navigator.fig8_crossings // 2
                    state_disp += f" ({cycles}/3 cycles)"
                elif navigator.state == "E_STOP":
                    state_disp += " (POWER CUT)"

                ax1.set_title(f"ManaTM | State: {state_disp}")

                # t1 = (transforms.Affine2D()
                #       .rotate(navigator.est_theta)
                #       .translate(navigator.est_x, navigator.est_y)
                #       + ax1.transData)
                # boat_rect1.set_transform(t1)

                # t2 = (transforms.Affine2D()
                #       .rotate(navigator.est_theta)
                #       .translate(navigator.est_x, navigator.est_y)
                #       + ax2.transData)
                # boat_rect2.set_transform(t2)

                # cos_a  = np.cos(navigator.est_theta + navigator.lidar_angles[::5])
                # sin_a  = np.sin(navigator.est_theta + navigator.lidar_angles[::5])
                # ranges = navigator.lidar_ranges[::5]
                # n = len(ranges)
                # rx = np.empty(n * 3); ry = np.empty(n * 3)
                # rx[0::3] = navigator.est_x;  ry[0::3] = navigator.est_y
                # rx[1::3] = navigator.est_x + ranges * cos_a
                # ry[1::3] = navigator.est_y + ranges * sin_a
                # rx[2::3] = np.nan;           ry[2::3] = np.nan
                # # lidar_lines.set_data(rx, ry)

                # slam_img.set_data(navigator.slam_map)

                # for state_key, (hx, hy) in navigator.history_dict.items():
                #     if len(hx) > 0:
                #         history_lines[state_key].set_data(hx, hy)

                # if navigator.tracked_buoy1:
                #     buoy1_pt.set_data([navigator.tracked_buoy1[0]],
                #                       [navigator.tracked_buoy1[1]])
                # if navigator.tracked_buoy2:
                #     buoy2_pt.set_data([navigator.tracked_buoy2[0]],
                #                       [navigator.tracked_buoy2[1]])

                lidar_polar.set_offsets(
                    np.c_[navigator.lidar_angles, navigator.lidar_ranges])

                plt.pause(0.001)

            step += 1

            if navigator.state == "MISSION_COMPLETE":
                print("Mission complete!")
                break

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C — shutting down...")

    finally:
        _running = False
        time.sleep(1.0)
        stop_all()
        set_motor(0.0)
        set_rudder(90)
        set_pump(False)
        pixels.fill(0)
        pixels.show()
        cap1.release()
        cap2.release()
        try:
            lidar.stop_event.set()
            time.sleep(0.5)
            lidar.reset()
        except Exception as e:
            print(f"[LIDAR] shutdown error: {e}")
        plt.ioff()
        plt.close('all')
        print("[MAIN] Clean shutdown complete.")


if __name__ == "__main__":
    main()
