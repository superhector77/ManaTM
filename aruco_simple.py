import cv2
import numpy as np
import math
import time

INCH_TO_M = 0.0254
FEET_TO_M = 0.3048

class ArucoLocator:
    def __init__(self):
        # State variables
        self.last_known_pose = (0.0, 0.0, 0.0) # X, Y, Yaw
        self.caps = []
        self.detector = None
        self.CAM_MATRIX = None
        self.DIST_COEFFS = None
        self.tag_map_3d = {}
        self.T_cam1_in_robot = None
        self.T_cam2_in_robot = None
        self.TAG_SIZE = 0.208        # Tag size in meters
        self.TAG_SPACNG = 0.498      # 49.8 cm in meters (Distance from origin to origin)


    def setup(self):
        """Initializes cameras, ArUco detector, and transforms."""
        print("[Locator] Initializing...")
        
        # 1. Init Detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        # 2. Camera Intrinsics (Using the 640x480 estimate)
        self.CAM_MATRIX = np.array([[699.1, 0, 320.0],
                                    [0, 699.1, 240.0],
                                    [0, 0, 1.0]], dtype=np.float32)
        self.DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

        # 3. Generate Tag Map (From previous script)
        self.tag_map_3d = self._generate_aruco_map()

        # 4. Robot-to-Camera Transforms (2 inches converted to meters)
        offset_m = 2.0 * 0.0254
        self.T_cam1_in_robot = self._create_transform(0.0, -offset_m, 0.0, -90.0)
        self.T_cam2_in_robot = self._create_transform(0.0, offset_m, 0.0, 90.0)

        # 5. Start Cameras
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1) # Adjust index if needed
        
        for cap in [cap1, cap2]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        self.caps = [cap1, cap2]
        
        # Give cameras a second to warm up
        time.sleep(1.0)
        print("[Locator] Ready.")

    def get_position(self):
        """Reads frames, calculates pose, returns (X, Y, Yaw)."""
        valid_poses = []

        # Read both cameras
        ret1, frame1 = self.caps[0].read()
        ret2, frame2 = self.caps[1].read()

        if ret1:
            pos, rot = self._process_frame(frame1)
            if pos is not None:
                r_pos, r_rot = self._get_robot_pose(pos, rot, self.T_cam1_in_robot)
                valid_poses.append((r_pos, r_rot))

        if ret2:
            pos, rot = self._process_frame(frame2)
            if pos is not None:
                r_pos, r_rot = self._get_robot_pose(pos, rot, self.T_cam2_in_robot)
                valid_poses.append((r_pos, r_rot))

        # If no tags seen, return the last known position
        if not valid_poses:
            return self.last_known_pose

        # If tags seen, calculate the new position
        # If both cameras see tags, average their X, Y. (For Yaw, we just take the first one for simplicity here)
        avg_x = sum(p[0][0] for p in valid_poses) / len(valid_poses)
        avg_y = sum(p[0][1] for p in valid_poses) / len(valid_poses)
        
        # Extract Yaw from the first valid rotation matrix
        # Yaw is rotation around Z axis. math.atan2(R10, R00)
        R = valid_poses[0][1]
        yaw = math.atan2(R[1, 0], R[0, 0])

        # Update state and return
        self.last_known_pose = (avg_x, avg_y, yaw)
        return self.last_known_pose

    def cleanup(self):
        """Releases hardware resources."""
        for cap in self.caps:
            cap.release()

    # --- PRIVATE HELPER METHODS ---
    def _process_frame(self, frame):
        """Internal method to run PnP on a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            obj_points, img_points = [], []
            for i, tag_id in enumerate(ids.flatten()):
                if tag_id in self.tag_map_3d:
                    obj_points.extend(self.tag_map_3d[tag_id])
                    img_points.extend(corners[i][0])

            if len(obj_points) >= 4:
                obj_pts = np.array(obj_points, dtype=np.float32)
                img_pts = np.array(img_points, dtype=np.float32)
                success, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, self.CAM_MATRIX, self.DIST_COEFFS)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    R_inv = np.linalg.inv(R)
                    cam_pos = -np.dot(R_inv, tvec)
                    return cam_pos, R_inv
        return None, None

    def _get_robot_pose(self, cam_pos, cam_rot, T_cam_in_robot):
        """Applies transform matrix."""
        T_cam_in_world = np.eye(4, dtype=np.float32)
        T_cam_in_world[:3, :3] = cam_rot
        T_cam_in_world[:3, 3] = cam_pos.flatten()
        
        T_robot_in_world = np.dot(T_cam_in_world, np.linalg.inv(T_cam_in_robot))
        return T_robot_in_world[:3, 3], T_robot_in_world[:3, :3]

    def _create_transform(self, tx, ty, tz, rz_deg):
        rz_rad = math.radians(rz_deg)
        return np.array([
            [math.cos(rz_rad), -math.sin(rz_rad), 0, tx],
            [math.sin(rz_rad),  math.cos(rz_rad), 0, ty],
            [               0,                 0, 1, tz],
            [               0,                 0, 0,  1]
        ], dtype=np.float32)

    def _generate_aruco_map(self):
        """Generates the 3D map from the previous step."""
        # Insert the mapping logic we wrote previously here...
        # For brevity, returning a dummy map in this snippet:
        return {1: np.array([[0,0,0], [0.2,0,0], [0.2,0,-0.2], [0,0,-0.2]], dtype=np.float32)}


    def _generate_aruco_map(self):
        # ==========================================
        # CONFIGURATION & CONVERSIONS
        # ==========================================

        # Starting positions (converted to meters)
        # Tag 1: (0, -43 in, 0)
        tag_1_start = np.array([8.0 * INCH_TO_M, -43.0 * INCH_TO_M, -10.5 * INCH_TO_M])
        
        # Tag 6: (8 ft, 0, 0)
        tag_6_start = np.array([10.0 * FEET_TO_M, -34.0 * INCH_TO_M, -10.5 * INCH_TO_M])
        
        tag_map_3d = {}

        # ==========================================
        # WALL 1: TAGS 1 TO 5
        # ==========================================
        for i in range(5):
            tag_id = i + 1  # IDs 1, 2, 3, 4, 5
            
            # Calculate bottom-left (bl) coordinate for this specific tag
            # Shifts the X coordinate by TAG_SPACNG (0.498m) for each tag
            bl_x = tag_1_start[0] + (i * self.TAG_SPACNG)
            bl_y = tag_1_start[1]
            bl_z = tag_1_start[2]
            
            # Build corners in OpenCV order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            tag_map_3d[tag_id] = np.array([
                [bl_x,        bl_y, bl_z - self.TAG_SIZE], # Top Left
                [bl_x + self.TAG_SIZE, bl_y, bl_z - self.TAG_SIZE], # Top Right
                [bl_x + self.TAG_SIZE, bl_y, bl_z],        # Bottom Right
                [bl_x,        bl_y, bl_z]         # Bottom Left
            ], dtype=np.float32)

        # ==========================================
        # WALL 2: TAGS 6 TO 9
        # ==========================================
        for i in range(4):
            tag_id = i + 6  # IDs 6, 7, 8, 9
            
            # Calculate bottom-left (bl) coordinate for this specific tag
            # Shifts the Y coordinate by TAG_SPACNG (0.498m) for each tag
            bl_x = tag_6_start[0]
            bl_y = tag_6_start[1] + (i * self.TAG_SPACNG)
            bl_z = tag_6_start[2]
            
            # Build corners in OpenCV order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            tag_map_3d[tag_id] = np.array([
                [bl_x, bl_y,        bl_z - self.TAG_SIZE], # Top Left
                [bl_x, bl_y + self.TAG_SIZE, bl_z - self.TAG_SIZE], # Top Right
                [bl_x, bl_y + self.TAG_SIZE, bl_z],        # Bottom Right
                [bl_x, bl_y,        bl_z]         # Bottom Left
            ], dtype=np.float32)

        return tag_map_3d

# ==========================================
# USAGE IN YOUR MAIN BOAT LOOP
# ==========================================
if __name__ == "__main__":
    boat_vision = ArucoLocator()
    boat_vision.setup()
    
    try:
        while True:
            # Look how clean your main loop is now!
            x, y, yaw = boat_vision.get_position()
            
            # Print with 3 decimal places
            print(f"X: {x:.3f}m | Y: {y:.3f}m | Yaw: {math.degrees(yaw):.1f}°")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        boat_vision.cleanup()