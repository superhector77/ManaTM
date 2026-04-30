# V6.5.2.5: Debugged for drifting buoys, added anti-aperture pillars, and tuned SLAM parameters for better drift correction. Still using the same thrust/angle to motor/servo conversion as before.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
import sys
import os

import math
import time

# ==========================================
# LOCAL LIBRARY IMPORT SETUP
# ==========================================
LOCAL_BREEZYSLAM_DIR = '.' 

if LOCAL_BREEZYSLAM_DIR not in sys.path:
    sys.path.insert(0, LOCAL_BREEZYSLAM_DIR)

try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import breezyslam locally.")
    print(f"Please ensure a folder named 'breezyslam' exists inside: {os.path.abspath(LOCAL_BREEZYSLAM_DIR)}")
    print(f"Exact error: {e}")
    exit(1)


# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
MAP_SIZE = 10.0          
GRID_RES = 0.05
GRID_DIMS = int(MAP_SIZE / GRID_RES)

BOAT_LENGTH = 0.762       
BOAT_WIDTH = 0.2
REAR_AXIS_OFFSET = 0.05   
MAX_SPEED = 1
MAX_STEER = np.radians(45)
DT = 0.1

BUOY1X = 0.7*MAP_SIZE
BUOY1Y = 0.6*MAP_SIZE
BUOY2X = 0.35*MAP_SIZE
BUOY2Y = 0.3*MAP_SIZE

# --- HARDWARE CONFIGURATIONS --- 
LIDAR_FOV = np.radians(360)
LIDAR_RAYS = 90
LIDAR_RANGE = 12.0 
LIDAR_MIN_RANGE = 0.05      

class TugboatLidar(Laser):
    def __init__(self):
        super().__init__(LIDAR_RAYS, int(1.0/DT), 360, int(LIDAR_RANGE * 1000))

# ==========================================
# ENVIRONMENT & GROUND TRUTH
# ==========================================
class TrueEnvironment:
    def __init__(self):
        self._base_grid = np.zeros((GRID_DIMS, GRID_DIMS))
        border = int(0.03 * GRID_DIMS)
        self._base_grid[0:border, :] = self._base_grid[-border:, :] = 1
        self._base_grid[:, 0:border] = self._base_grid[:, -border:] = 1
        
        # # --- NEW: ANTI-APERTURE PILLARS ---
        # # Added distinct static structures so SLAM can detect lateral sideways slip!
        # self._base_grid[int(0.2*GRID_DIMS):int(0.3*GRID_DIMS), int(0.1*GRID_DIMS):int(0.15*GRID_DIMS)] = 1
        # self._base_grid[int(0.7*GRID_DIMS):int(0.8*GRID_DIMS), int(0.85*GRID_DIMS):int(0.9*GRID_DIMS)] = 1
        
        self.tether_b1 = np.array([BUOY1X, BUOY1Y])
        self.tether_b2 = np.array([BUOY2X, BUOY2Y])
        
        self.true_buoy1 = self.tether_b1.copy()
        self.true_buoy2 = self.tether_b2.copy()
        
        self.drift_radius = 0.5
        self.drift_speed = 0.1 
        
        self.grid = self._base_grid.copy()
        self.update_drift(0.0)

    def update_drift(self, dt):
        if dt > 0:
            self.drift_radius = 0.3
            self.drift_speed = 0.8  
            spring_strength = 2.0   
            
            for pos, tether in [(self.true_buoy1, self.tether_b1), (self.true_buoy2, self.tether_b2)]:
                angle = np.random.uniform(0, 2 * np.pi)
                pos[0] += np.cos(angle) * self.drift_speed * dt
                pos[1] += np.sin(angle) * self.drift_speed * dt
                
                pos[0] += (tether[0] - pos[0]) * spring_strength * dt
                pos[1] += (tether[1] - pos[1]) * spring_strength * dt

                dist = np.hypot(pos[0] - tether[0], pos[1] - tether[1])
                if dist > self.drift_radius:
                    pos[0] = tether[0] + self.drift_radius * (pos[0] - tether[0]) / dist
                    pos[1] = tether[1] + self.drift_radius * (pos[1] - tether[1]) / dist

        self.grid = self._base_grid.copy()
        for pos in [self.true_buoy1, self.true_buoy2]:
            bx = int(round(pos[0] / GRID_RES))
            by = int(round(pos[1] / GRID_RES))
            self.grid[max(0, by-2):min(GRID_DIMS, by+3), max(0, bx-2):min(GRID_DIMS, bx+3)] = 1

    def is_occupied(self, x, y):
        gx, gy = int(x / GRID_RES), int(y / GRID_RES)
        if 0 <= gx < GRID_DIMS and 0 <= gy < GRID_DIMS:
            return self.grid[gy, gx] == 1
        return True

    def check_collision(self, x, y, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        L_f = BOAT_LENGTH - REAR_AXIS_OFFSET
        L_r = -REAR_AXIS_OFFSET
        W_2 = BOAT_WIDTH / 2
        
        pts = [(0, 0), (L_f, W_2), (L_f, -W_2), (L_r, W_2), (L_r, -W_2)]
        for px, py in pts:
            gx = x + px*cos_t - py*sin_t
            gy = y + px*sin_t + py*cos_t
            if self.is_occupied(gx, gy):
                cell_cx = int(gx / GRID_RES) * GRID_RES + (GRID_RES / 2)
                cell_cy = int(gy / GRID_RES) * GRID_RES + (GRID_RES / 2)
                return True, cell_cx, cell_cy
        return False, 0.0, 0.0

# ==========================================
# HARDWARE ABSTRACTION SIMULATORS
# ==========================================
class RPLidarC1Simulator:
    def __init__(self, fov, rays, max_range, min_range):
        self.fov = fov
        self.angles = np.linspace(-fov/2, fov/2, rays)
        self.max_range = max_range
        self.min_range = min_range
        self.dropout_prob = 0.02   
        self.noise_base = 0.01     
        self.noise_prop = 0.005    
        
    def get_scan(self, env, true_x, true_y, true_theta):
        scan_matrix = []
        grid = env.grid
        step_sz = GRID_RES / 2.0
        
        cos_vals = np.cos(true_theta + self.angles)
        sin_vals = np.sin(true_theta + self.angles)
        r_array = np.arange(0, self.max_range, step_sz)
        
        for i, angle in enumerate(self.angles):
            angle_deg = np.degrees(angle)
            if np.random.random() < self.dropout_prob:
                scan_matrix.append([angle_deg, None])
                continue

            hit = False
            cx, cy = cos_vals[i], sin_vals[i]
            
            for r in r_array:
                gx = int((true_x + r * cx) / GRID_RES)
                gy = int((true_y + r * cy) / GRID_RES)
                
                if 0 <= gx < GRID_DIMS and 0 <= gy < GRID_DIMS:
                    if grid[gy, gx] == 1:
                        if r < self.min_range:
                            scan_matrix.append([angle_deg, None])
                        else:
                            noise = np.random.normal(0, self.noise_base + (r * self.noise_prop))
                            dist_m = min(r + noise, self.max_range)
                            scan_matrix.append([angle_deg, dist_m])
                        hit = True
                        break
                else:
                    if r < self.min_range:
                        scan_matrix.append([angle_deg, None])
                    else:
                        noise = np.random.normal(0, self.noise_base + (r * self.noise_prop))
                        dist_m = min(r + noise, self.max_range)
                        scan_matrix.append([angle_deg, dist_m])
                    hit = True
                    break
                    
            if not hit:
                scan_matrix.append([angle_deg, None])
                
        return scan_matrix

class SensorData:
    def __init__(self, x, y, z):
        self.xData = x
        self.yData = y
        self.zData = z

class BoschBNO055Simulator:
    def __init__(self):
        self.gyro_noise_std = 0.02    
        self.accel_noise_std = 0.05

    def read_imu(self, true_ax, true_ay, true_yaw_rate):
        noise_g = np.random.normal(0, self.gyro_noise_std)
        noise_ax = np.random.normal(0, self.accel_noise_std)
        noise_ay = np.random.normal(0, self.accel_noise_std)
        
        accelData = SensorData(true_ax + noise_ax, true_ay + noise_ay, 0.0)
        gyroData = SensorData(0.0, 0.0, true_yaw_rate + noise_g)
        
        accel_tuple = (accelData.xData, accelData.yData, accelData.zData)
        gyro_tuple = (gyroData.xData, gyroData.yData, gyroData.zData)
        
        return accel_tuple, gyro_tuple

# ==========================================
# DIGITAL TWIN
# ==========================================
class AdvancedDigitalTwin:
    def __init__(self, start_pos=(0.15*MAP_SIZE, 0.9*MAP_SIZE, -np.pi/4)):
        self._x, self._y, self._theta = start_pos
        self._v = 0.0
        
        self.lidar_hw = RPLidarC1Simulator(fov=LIDAR_FOV, rays=LIDAR_RAYS, 
                                           max_range=LIDAR_RANGE, min_range=LIDAR_MIN_RANGE)
        self.imu_hw = BoschBNO055Simulator()
        
        self.laser_spec = TugboatLidar()
        
        # --- NEW TUNED SLAM PARAMETERS ---
        # sigma_xy_mm (150): Allows SLAM to search heavily for lateral drift matches.
        # sigma_theta_degrees (1): Forces SLAM to strictly trust the IMU Gyro to prevent map rotation.
        self.slam = RMHC_SLAM(self.laser_spec, GRID_DIMS, int(MAP_SIZE), 
                              sigma_xy_mm=150, sigma_theta_degrees=1, max_search_iter=2000, random_seed=42)
        
        self.slam.position.x_mm = start_pos[0] * 1000.0
        self.slam.position.y_mm = start_pos[1] * 1000.0
        self.slam.position.theta_degrees = np.degrees(start_pos[2])
        
        self.est_x, self.est_y, self.est_theta = start_pos 
        self.lidar_ranges = np.zeros(LIDAR_RAYS)
        self.lidar_angles = np.linspace(-LIDAR_FOV/2, LIDAR_FOV/2, LIDAR_RAYS)
        self.slam_map = np.zeros((GRID_DIMS, GRID_DIMS))
        
        self.history_dict = {
            "LEAVE_DOCK1": ([], []), "WANDER": ([], []), "SEARCH_BUOY1": ([], []),
            "CIRCLE_BUOY1": ([], []), "APPROACH_DOCK1": ([], []), "CIRCLE_DOCK1": ([], []),
            "FIND_BUOYS": ([], []), "FIGURE_8": ([], []), "APPROACH_DOCK2": ([], []),
            "CIRCLE_DOCK2": ([], []), "RETURN_TO_BASE": ([], []), "E_STOP": ([], [])
        }

        self.collision_count = 0
        self.collision_cooldown = 0
        self.col_force_x = 0.0
        self.col_force_y = 0.0

        self.mass = 10.0       
        self.I_z = 2.0         
        
        self.u = 0.0           
        self.v = 0.0           
        self.r = 0.0 
        
        self.dot_u = 0.0
        self.dot_v = 0.0          
        
        self.X_u, self.X_uu = 2.0, 5.0     
        self.Y_v, self.Y_vv = 15.0, 30.0   
        self.N_r, self.N_rr = 3.0, 8.0     
        
        self.kp_thrust = 20.0
        self.ki_thrust = 2.0
        
        self.kp_steer = 0.8 # 10.0   
        self.ki_steer = 0.05 # 0.5    
        self.kd_steer = 0.5 # 8.0   
        
        self.integral_heading_error = 0.0
        self.integral_speed_error = 0.0
        
        self.current_thrust = 0.0
        self.current_moment = 0.0
        
        self.start_dock = (start_pos[0], start_pos[1])
        self.end_dock = (0.9*int(MAP_SIZE),0.1*int(MAP_SIZE))
        self.expected_buoy1 = (BUOY1X, BUOY1Y)
        self.expected_buoy2 = (BUOY2X, BUOY2Y)
        self.rtb_target = (0.2*int(MAP_SIZE), 0.8*int(MAP_SIZE))

        clearance_x = self.start_dock[0] + 0.5 * np.cos(start_pos[2])
        clearance_y = self.start_dock[1] + 0.5 * np.sin(start_pos[2])
        
        self.path_leave_dock1 = [(clearance_x, clearance_y)]
        self.path_approach_dock1 = [self.rtb_target, self.start_dock]
        self.path_approach_dock2 = [(self.end_dock[0], self.end_dock[1] + 0.1*MAP_SIZE), self.end_dock]
        
        self.state = "LEAVE_DOCK1"
        self.wander_angle = start_pos[2]
        self.sim_time = 0.0
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

    def sense(self, env):
        lidar_matrix = self.lidar_hw.get_scan(env, self._x, self._y, self._theta)
        accel_tuple, gyro_tuple = self.imu_hw.read_imu(self.dot_u, self.dot_v, self.r)
        
        is_hit, hit_x, hit_y = env.check_collision(self._x, self._y, self._theta)
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

        if is_hit:
            if self.collision_cooldown == 0:
                self.collision_count += 1
                self.collision_cooldown = 10 
            
            self.col_force_x = 0.0
            self.col_force_y = 0.0
            self.u = 0.0
            self.v = 0.0
            self.r = 0.0
        else:
            self.col_force_x = 0.0
            self.col_force_y = 0.0
            
        return lidar_matrix, accel_tuple, gyro_tuple

    def get_buoy_update(self, tracked_pos, expected_pos):
        ref_pos = tracked_pos if tracked_pos else expected_pos
        best_idx = -1
        min_err = float('inf')
        
        for i, r in enumerate(self.lidar_ranges):
            if r < LIDAR_RANGE:
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

    def think(self, lidar_matrix, accel_tuple, gyro_tuple):
        ax, ay, az = accel_tuple
        gx, gy, gz = gyro_tuple

        # --- 1. SENSOR FUSION & SLAM ESTIMATION ---
        scan_mm = []
        for i, row in enumerate(lidar_matrix):
            angle_deg, dist_m = row
            if dist_m is None:
                scan_mm.append(0)
                self.lidar_ranges[i] = LIDAR_RANGE
            else:
                scan_mm.append(int(dist_m * 1000.0))
                self.lidar_ranges[i] = dist_m

        # --- FIX: STRICT UNICYCLE ODOMETRY ---
        # The unicycle model assumes 0 sideways slip. By passing self.u instead of hypot(),
        # we explicitly tell SLAM "we drove forward X distance". When the LIDAR sees the map 
        # move sideways, it now correctly identifies it as unrecorded lateral slip and snaps the map!
        dxy_mm = self.u * 1000.0 * DT
        dtheta_degrees = np.degrees(gz) * DT

        # Run the actual SLAM algorithm using pose_change
        self.slam.update(scan_mm, pose_change=(dxy_mm, dtheta_degrees, DT))

        x_mm, y_mm, theta_deg = self.slam.getpos()
        self.est_x = x_mm / 1000.0
        self.est_y = y_mm / 1000.0
        self.est_theta = np.radians(theta_deg)

        mapbytes = bytearray(GRID_DIMS * GRID_DIMS)
        self.slam.getmap(mapbytes)
        self.slam_map = np.array(mapbytes).reshape((GRID_DIMS, GRID_DIMS))
        
        # --- 2. COGNITIVE MISSION STATE MACHINE ---
        if self.state in ["MISSION_COMPLETE", "E_STOP"]:
            self.current_thrust = 0.0
            self.current_moment = 0.0
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
            dx = self.rtb_target[0] - self.est_x
            dy = self.rtb_target[1] - self.est_y
            dist = np.hypot(dx, dy)
            angle_to_rtb = np.arctan2(dy, dx)
            if dist > 1.5:
                target_v, target_heading = self._arbiter(angle_to_rtb, MAX_SPEED)
            else:
                target_radius = 1.0
                direction = -1 # CW
                K = 1.5
                alpha = np.arctan(K * (dist - target_radius))
                tangent_offset = direction * (np.pi/2 - alpha)
                orbital_heading = angle_to_rtb + tangent_offset
                target_v, target_heading = self._arbiter(orbital_heading, 0.5)

        elif self.state == "SEARCH_BUOY1":
            self.search_angle += 0.05
            search_radius = 0.5 + (self.search_angle * 0.05)
            sx = self.expected_buoy1[0] + search_radius * np.cos(self.search_angle)
            sy = self.expected_buoy1[1] + search_radius * np.sin(self.search_angle)
            
            pp_heading = np.arctan2(sy - self.est_y, sx - self.est_x)
            target_v, target_heading = self._arbiter(pp_heading, MAX_SPEED * 0.7)

            differences = np.abs(np.diff(self.lidar_ranges))
            if np.any((differences > 1.0) & (self.lidar_ranges[:-1] < 3)):
                self.state = "CIRCLE_BUOY1"
                self.tracked_buoy1, _, _ = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
                self.total_circled_angle = 0.0
                self.last_buoy_angle = np.arctan2(self.est_y - self.tracked_buoy1[1], self.est_x - self.tracked_buoy1[0])

        elif self.state == "CIRCLE_BUOY1":
            self.tracked_buoy1, _, _ = self.get_buoy_update(self.tracked_buoy1, self.expected_buoy1)
            target_radius = 1.5  
            direction = -1 # CW
            
            dx = self.tracked_buoy1[0] - self.est_x
            dy = self.tracked_buoy1[1] - self.est_y
            d = np.hypot(dx, dy)
            angle_to_buoy = np.arctan2(dy, dx)
            
            K = 1.5
            alpha = np.arctan(K * (d - target_radius))
            
            tangent_offset = direction * (np.pi/2 - alpha)
            orbital_heading = angle_to_buoy + tangent_offset
            
            target_v, target_heading = self._arbiter(orbital_heading, 0.7, ignore_target=self.tracked_buoy1)
            
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
            dx = target[0] - self.est_x
            dy = target[1] - self.est_y
            d = np.hypot(dx, dy)
            angle_to_dock = np.arctan2(dy, dx)
            
            target_radius = 1.5  
            direction = -1 # CW
            K = 1.5
            alpha = np.arctan(K * (d - target_radius))
            tangent_offset = direction * (np.pi/2 - alpha)
            orbital_heading = angle_to_dock + tangent_offset
            
            target_v, target_heading = self._arbiter(orbital_heading, 0.5)

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
                
                dx = active_target[0] - self.est_x
                dy = active_target[1] - self.est_y
                d = np.hypot(dx, dy)
                angle_to_buoy = np.arctan2(dy, dx)
                
                K = 1.5
                alpha = np.arctan(K * (d - target_radius))
                tangent_offset = direction * (np.pi/2 - alpha)
                orbital_heading = angle_to_buoy + tangent_offset
                
                target_v, target_heading = self._arbiter(orbital_heading, 0.7, ignore_target=active_target)

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
        
        self.integral_heading_error += heading_error * DT
        self.integral_heading_error = np.clip(self.integral_heading_error, -10.0, 10.0) 
        
        self.integral_speed_error += speed_error * DT
        self.integral_speed_error = np.clip(self.integral_speed_error, -20.0, 20.0)
        
        MAX_THRUST = 40.0
        thrust_cmd = np.clip((self.kp_thrust * speed_error) + (self.ki_thrust * self.integral_speed_error), 0.0, MAX_THRUST)
        
        # Calculate Propeller Angle in Radians
        prop_angle_cmd = np.clip(
            (self.kp_steer * heading_error) + 
            (self.ki_steer * self.integral_heading_error) - 
            (self.kd_steer * gz), 
            -MAX_STEER, MAX_STEER
        )

        self.current_thrust = thrust_cmd
        # (Optional) Update this tracking variable so the UI visualizer works correctly
        self.current_moment = prop_angle_cmd 
        
        return thrust_cmd, prop_angle_cmd

    def _arbiter(self, desired_heading, max_speed, ignore_target=None):
        hist_bins = 72 
        histogram = np.zeros(hist_bins)
        
        SAFE_DIST = 3.0 
        
        valid_ranges = np.copy(self.lidar_ranges)
        if ignore_target is not None:
            for i, r in enumerate(self.lidar_ranges):
                if r < SAFE_DIST:
                    ray_ang = self.est_theta + self.lidar_angles[i]
                    rx = self.est_x + r * np.cos(ray_ang)
                    ry = self.est_y + r * np.sin(ray_ang)
                    
                    if np.hypot(rx - ignore_target[0], ry - ignore_target[1]) < 1.0:
                        valid_ranges[i] = LIDAR_RANGE
        
        min_dist = np.min(valid_ranges)
        
        MIN_SPEED = 0.2
        if min_dist < 0.6:
            left_openness = np.mean([r for i, r in enumerate(valid_ranges) if self.lidar_angles[i] > 0])
            right_openness = np.mean([r for i, r in enumerate(valid_ranges) if self.lidar_angles[i] < 0])
            escape_dir = 1.0 if left_openness > right_openness else -1.0
            return MIN_SPEED, self.est_theta + (escape_dir * np.pi/2)

        for i, r in enumerate(valid_ranges):
            if r < SAFE_DIST:
                global_angle = (self.est_theta + self.lidar_angles[i]) % (2*np.pi)
                bin_idx = int((global_angle / (2*np.pi)) * hist_bins)
                block_width = int(2 + (SAFE_DIST - r) * 3) 
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
        
        path_clearance = 4.0 
        for i, r in enumerate(valid_ranges):
            ray_global_angle = (self.est_theta + self.lidar_angles[i]) % (2*np.pi)
            if ang_dist(ray_global_angle, best_heading) < np.pi/6: 
                if r < path_clearance:
                    path_clearance = r
                    
        LOOKAHEAD, CRITICAL, MIN_SPEED_FACTOR = 3.0, 0.6, 0.3
        if path_clearance >= LOOKAHEAD:
            path_speed_factor = 1.0 
        elif path_clearance <= CRITICAL:
            path_speed_factor = MIN_SPEED_FACTOR 
        else:
            ratio = (path_clearance - CRITICAL) / (LOOKAHEAD - CRITICAL)
            path_speed_factor = MIN_SPEED_FACTOR + (1.0 - MIN_SPEED_FACTOR) * ratio
            
        turn_severity = ang_dist(best_heading, desired_wrapped)
        turn_speed_factor = max(0.6, 1.0 - (turn_severity / np.pi))
        
        final_speed = max_speed * path_speed_factor * turn_speed_factor
        return max(MIN_SPEED if max_speed > 0 else 0.0, final_speed), best_heading

    def act(self, thrust_cmd, prop_angle_cmd):
        if self.state in ["MISSION_COMPLETE", "E_STOP"]:
            thrust_cmd = 0.0
            prop_angle_cmd = 0.0
            
        self.sim_time += DT 
        cos_t, sin_t = np.cos(self._theta), np.sin(self._theta)
        col_surge = self.col_force_x * cos_t + self.col_force_y * sin_t
        col_sway = -self.col_force_x * sin_t + self.col_force_y * cos_t

        # --- NEW: Steerable Thruster Vectoring Physics ---
        # The propeller is angled, so thrust is split into X and Y forces relative to the boat.
        prop_force_x = thrust_cmd * np.cos(prop_angle_cmd)
        prop_force_y = thrust_cmd * np.sin(prop_angle_cmd)
        
        # Calculate the actual moment (torque) generated by the sideways push.
        # Assuming the prop is located at the back of the boat: Moment = Force * Lever_Arm
        lever_arm = BOAT_LENGTH / 2.0
        generated_moment = prop_force_y * lever_arm
        # -------------------------------------------------

        drag_X = -(self.X_u * self.u + self.X_uu * self.u * abs(self.u))
        drag_Y = -(self.Y_v * self.v + self.Y_vv * self.v * abs(self.v))
        drag_N = -(self.N_r * self.r + self.N_rr * self.r * abs(self.r))
        
        # Apply the vectored forces and the resulting moment to the boat's kinematics
        self.dot_u = (prop_force_x + drag_X + col_surge + (self.mass * self.v * self.r)) / self.mass
        self.dot_v = (prop_force_y + drag_Y + col_sway - (self.mass * self.u * self.r)) / self.mass
        dot_r = (generated_moment + drag_N) / self.I_z
        
        self.u += self.dot_u * DT
        self.v += self.dot_v * DT
        self.r += dot_r * DT
        
        self.u = np.clip(self.u, -5.0, 5.0)
        self.v = np.clip(self.v, -5.0, 5.0)
        self.r = np.clip(self.r, -5.0, 5.0)
        
        global_vx = self.u * np.cos(self._theta) - self.v * np.sin(self._theta)
        global_vy = self.u * np.sin(self._theta) + self.v * np.cos(self._theta)
        
        self._x += global_vx * DT
        self._y += global_vy * DT
        self._theta += self.r * DT
        self._theta = (self._theta + np.pi) % (2 * np.pi) - np.pi
        self._v = np.hypot(global_vx, global_vy)

        if self.state in self.history_dict:
            self.history_dict[self.state][0].append(self.est_x)
            self.history_dict[self.state][1].append(self.est_y)

# ==========================================
# VISUALIZATION LOOP
# ==========================================
def draw_boat(ax, x, y, theta, color='blue', alpha=1.0):
    rect = patches.Rectangle((-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH, 
                             linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
    t = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

def main():
    env = TrueEnvironment()
    boat = AdvancedDigitalTwin()
    
    plt.ion()
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='polar') 
    plt.subplots_adjust(bottom=0.15) 
    
    ax_start = plt.axes([0.16, 0.02, 0.15, 0.06])
    btn_start = Button(ax_start, 'START M1: CIRCLE', color='forestgreen', hovercolor='limegreen')
    btn_start.label.set_color('white')
    btn_start.label.set_fontweight('bold')

    ax_pause = plt.axes([0.33, 0.02, 0.08, 0.06])
    btn_pause = Button(ax_pause, 'PAUSE', color='gold', hovercolor='yellow')
    btn_pause.label.set_fontweight('bold')

    ax_home = plt.axes([0.43, 0.02, 0.11, 0.06]) 
    btn_home = Button(ax_home, 'RETURN HOME', color='darkorange', hovercolor='orange')
    btn_home.label.set_color('white')
    btn_home.label.set_fontweight('bold')
    
    ax_estop = plt.axes([0.56, 0.02, 0.08, 0.06]) 
    btn_estop = Button(ax_estop, 'E-STOP', color='red', hovercolor='darkred')
    btn_estop.label.set_color('white')
    btn_estop.label.set_fontweight('bold')
    
    mission_stage = [1]
    def start_callback(event):
        if mission_stage[0] == 1:
            boat.state = "SEARCH_BUOY1"
            boat.wp_index = 0
            btn_start.label.set_text('START M2: FIG-8')
            mission_stage[0] = 2
            print("\n[>] MISSION 1 STARTED: Circling Buoy 1.")
        elif mission_stage[0] == 2:
            boat.state = "FIND_BUOYS"
            boat.wp_index = 0
            btn_start.label.set_text('MISSIONS ACTIVE')
            btn_start.color = 'gray'
            mission_stage[0] = 3
            print("\n[>] MISSION 2 STARTED: Figure-Eight.")
    btn_start.on_clicked(start_callback)

    pause_state = {"paused": False}
    def pause_callback(event):
        pause_state["paused"] = not pause_state["paused"]
        btn_pause.label.set_text("RESUME" if pause_state["paused"] else "PAUSE")
    btn_pause.on_clicked(pause_callback)

    def home_callback(event):
        boat.state = "RETURN_TO_BASE"
        print("\n[<] RETURN HOME TRIGGERED: Plotting safe course to base.")
    btn_home.on_clicked(home_callback)
    
    def estop_callback(event):
        boat.state = "E_STOP"
        print("\n[!] E-STOP TRIGGERED: Power completely cut. Coasting...")
    btn_estop.on_clicked(estop_callback)

    ax_throttle = plt.axes([0.08, 0.03, 0.06, 0.04]) 
    ax_throttle.set_xlim(0, 50) 
    ax_throttle.set_yticks([])
    ax_throttle.set_title("Throttle (N)", fontsize=10)
    throttle_bar = ax_throttle.barh([0], [0], color='limegreen')

    ax_rudder = plt.axes([0.66, 0.03, 0.12, 0.04])
    ax_rudder.set_xlim(-20, 20) 
    ax_rudder.axvline(0, color='black', linewidth=1) 
    ax_rudder.set_yticks([])
    ax_rudder.set_title("Rudder Moment (Nm)", fontsize=10)
    rudder_bar = ax_rudder.barh([0], [0], color='dodgerblue')

    ax1_title = ax1.set_title("")
    env_img = ax1.imshow(env.grid, origin='lower', cmap='Greys', extent=[0, MAP_SIZE, 0, MAP_SIZE])
    
    rtb_circle = patches.Circle(boat.rtb_target, 0.5, color='orange', fill=False, linestyle='--', alpha=0.6)
    ax1.add_patch(rtb_circle)
    ax1.plot(boat.rtb_target[0], boat.rtb_target[1], 'o', color='orange', markersize=4, label="RTB Point")
    ax1.plot(boat.end_dock[0], boat.end_dock[1], 'rs', markersize=12, label="Dock 2") 
    ax1.plot(boat.start_dock[0], boat.start_dock[1], 'gs', markersize=12, label="Dock 1") 

    boat_true_rect = patches.Rectangle((-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH, 
                                       linewidth=1, edgecolor='black', facecolor='cyan', alpha=1.0)
    ax1.add_patch(boat_true_rect)
    boat_est_rect = patches.Rectangle((-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH, 
                                      linewidth=1, edgecolor='black', facecolor='orange', alpha=0.5)
    ax1.add_patch(boat_est_rect)
    
    lidar_lines, = ax1.plot([], [], color='red', alpha=0.3, linewidth=0.5)
    collision_star, = ax1.plot([], [], marker='*', color='yellow', markersize=20, markeredgecolor='red', linestyle='None')
    
    ax2_title = ax2.set_title("BreezySLAM Generated Map")
    slam_img = ax2.imshow(np.zeros((GRID_DIMS, GRID_DIMS)), origin='lower', cmap='Blues', extent=[0, MAP_SIZE, 0, MAP_SIZE], vmin=0, vmax=255)
    
    maneuver_colors = {
        "LEAVE_DOCK1": "green", "WANDER": "gray", "SEARCH_BUOY1": "purple",
        "CIRCLE_BUOY1": "cyan", "APPROACH_DOCK1": "green", "CIRCLE_DOCK1": "green",
        "FIND_BUOYS": "purple", "FIGURE_8": "magenta", "APPROACH_DOCK2": "red",
        "CIRCLE_DOCK2": "red", "RETURN_TO_BASE": "orange", "E_STOP": "black"
    }
    history_lines = {}
    for state, color in maneuver_colors.items():
        line, = ax2.plot([], [], color=color, marker='.', linestyle='None', markersize=3, alpha=0.6, zorder=2)
        history_lines[state] = line

    boat_est_rect2 = patches.Rectangle((-REAR_AXIS_OFFSET, -BOAT_WIDTH/2), BOAT_LENGTH, BOAT_WIDTH, 
                                       linewidth=1, edgecolor='black', facecolor='orange', alpha=1.0)
    ax2.add_patch(boat_est_rect2)
    
    buoy1_exp_point, = ax2.plot(boat.expected_buoy1[0], boat.expected_buoy1[1], 'yx', markersize=10)
    buoy2_exp_point, = ax2.plot(boat.expected_buoy2[0], boat.expected_buoy2[1], 'yx', markersize=10)
    buoy1_trk_point, = ax2.plot([], [], 'go', markersize=8)
    buoy2_trk_point, = ax2.plot([], [], 'mo', markersize=8)

    ax3.set_title("LIDAR Sensor Data (Boat Frame)")
    ax3.set_theta_zero_location('N') 
    ax3.set_theta_direction(1)       
    ax3.set_ylim(0, LIDAR_RANGE)
    ax3.set_yticks(np.arange(1, LIDAR_RANGE + 1, 1.0))
    ax3.tick_params(axis='y', labelsize=8)
    lidar_polar = ax3.scatter(boat.lidar_angles, boat.lidar_ranges, c='red', s=15, alpha=0.8)
    ax3.plot(0, 0, marker='^', color='blue', markersize=12)

    ax1.set_xlim(0, MAP_SIZE); ax1.set_ylim(0, MAP_SIZE)
    ax2.set_xlim(0, MAP_SIZE); ax2.set_ylim(0, MAP_SIZE)

    step = 0
    while True:
        if pause_state["paused"]:
            plt.pause(0.05)
            continue

        env.update_drift(DT) 
        
        lidar_matrix, accel_tuple, gyro_tuple = boat.sense(env)
        cmd_thrust, cmd_rudder = boat.think(lidar_matrix, accel_tuple, gyro_tuple)
        boat.act(cmd_thrust, cmd_rudder)

        if step % 5 == 0:   
            throttle_bar[0].set_width(boat.current_thrust)
            throttle_bar[0].set_color('red' if boat.current_thrust > 45 else 'limegreen')
            rudder_bar[0].set_width(boat.current_moment)
            
            state_disp = boat.state
            if boat.state == "CIRCLE_BUOY1":
                loops = int(abs(boat.total_circled_angle) / (2*np.pi))
                state_disp += f" ({loops}/3 loops)"
            elif boat.state in ["CIRCLE_DOCK1", "CIRCLE_DOCK2"]:
                state_disp += " (Holding position)"
            elif boat.state == "FIGURE_8":
                cycles = boat.fig8_crossings // 2
                state_disp += f" ({cycles}/3 cycles)"
            elif boat.state == "RETURN_TO_BASE":
                state_disp += " (Holding safe orbit)"
            elif boat.state == "E_STOP":
                state_disp += " (POWER CUT)"
                
            ax1_title.set_text(f"Real World (State: {state_disp}) | Collisions: {boat.collision_count}")
            env_img.set_data(env.grid)
            
            t_true = transforms.Affine2D().rotate(boat._theta).translate(boat._x, boat._y) + ax1.transData
            boat_true_rect.set_transform(t_true)
            
            t_est = transforms.Affine2D().rotate(boat.est_theta).translate(boat.est_x, boat.est_y) + ax1.transData
            boat_est_rect.set_transform(t_est)
            
            cos_angs = np.cos(boat._theta + boat.lidar_angles[::3])
            sin_angs = np.sin(boat._theta + boat.lidar_angles[::3])
            ranges = boat.lidar_ranges[::3]
            
            num_rays = len(ranges)
            rx = np.empty(num_rays * 3)
            ry = np.empty(num_rays * 3)
            rx[0::3] = boat._x
            ry[0::3] = boat._y
            rx[1::3] = boat._x + ranges * cos_angs
            ry[1::3] = boat._y + ranges * sin_angs
            rx[2::3] = np.nan
            ry[2::3] = np.nan
            lidar_lines.set_data(rx, ry)

            if boat.collision_cooldown > 5:
                collision_star.set_data([boat._x], [boat._y])
            else:
                collision_star.set_data([], [])

            slam_img.set_data(boat.slam_map)
            
            for state_key, (hx, hy) in boat.history_dict.items():
                if len(hx) > 0:
                    history_lines[state_key].set_data(hx, hy)

            t_est2 = transforms.Affine2D().rotate(boat.est_theta).translate(boat.est_x, boat.est_y) + ax2.transData
            boat_est_rect2.set_transform(t_est2)
            
            if boat.tracked_buoy1: buoy1_trk_point.set_data([boat.tracked_buoy1[0]], [boat.tracked_buoy1[1]])
            if boat.tracked_buoy2: buoy2_trk_point.set_data([boat.tracked_buoy2[0]], [boat.tracked_buoy2[1]])

            lidar_polar.set_offsets(np.c_[boat.lidar_angles, boat.lidar_ranges])

            plt.pause(0.001)
            
        step += 1
        
        if boat.state == "MISSION_COMPLETE" and step % 5 == 1:
            break
        
    print(f"Simulation Ended. Final State: {boat.state} | Total Collisions: {boat.collision_count}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()