import cv2
import numpy as np
import math

# ==========================================
# 1. CONFIGURATION & CALIBRATION
# ==========================================

INCH_TO_M = 0.0254
FEET_TO_M = 0.3048

# Use the smallest dictionary possible for faster detection
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Theoretical Intrinsics for eMeet C950 at 640x480
CAM_MATRIX = np.array([
    [699.1, 0,     320.0],
    [0,     699.1, 240.0],
    [0,     0,     1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

TAG_SIZE = 0.208        # Tag size in meters
TAG_SPACNG = 0.498      # 49.8 cm in meters (Distance from origin to origin)

# ==========================================
# 2. WORLD MAP DEFINITION
# ==========================================
# Define the 3D coordinates of the 4 corners of each tag in your world frame.
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# Format: { TagID : np.array([[x,y,z], [x,y,z], [x,y,z], [x,y,z]]) }
def generate_aruco_map():
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
        bl_x = tag_1_start[0] + (i * TAG_SPACNG)
        bl_y = tag_1_start[1]
        bl_z = tag_1_start[2]
        
        # Build corners in OpenCV order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        tag_map_3d[tag_id] = np.array([
            [bl_x,        bl_y, bl_z - TAG_SIZE], # Top Left
            [bl_x + TAG_SIZE, bl_y, bl_z - TAG_SIZE], # Top Right
            [bl_x + TAG_SIZE, bl_y, bl_z],        # Bottom Right
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
        bl_y = tag_6_start[1] + (i * TAG_SPACNG)
        bl_z = tag_6_start[2]
        
        # Build corners in OpenCV order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        tag_map_3d[tag_id] = np.array([
            [bl_x, bl_y,        bl_z - TAG_SIZE], # Top Left
            [bl_x, bl_y + TAG_SIZE, bl_z - TAG_SIZE], # Top Right
            [bl_x, bl_y + TAG_SIZE, bl_z],        # Bottom Right
            [bl_x, bl_y,        bl_z]         # Bottom Left
        ], dtype=np.float32)

    return tag_map_3d

tag_map_3d = generate_aruco_map()

# ==========================================
# 3. POSE ESTIMATION LOGIC
# ==========================================

def get_camera_pose_in_world(rvec, tvec):
    """
    solvePnP gives the World-to-Camera transform. 
    We invert it to get the Camera's position in the World frame.
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Calculate camera position in world: P_cam = -R_inv * tvec
    R_inv = np.linalg.inv(R)
    camera_position = -np.dot(R_inv, tvec)
    
    return camera_position, R_inv

def process_frame(frame, camera_name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        obj_points = [] # 3D world points
        img_points = [] # 2D image points

        # Match detected tags to our 3D map
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id in tag_map_3d:
                obj_points.extend(tag_map_3d[tag_id])
                img_points.extend(corners[i][0])

        if len(obj_points) >= 4: # Need at least 4 points (1 tag) for solvePnP
            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = np.array(img_points, dtype=np.float32)

            # solvePnPRansac is more robust to false corner detections than solvePnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_points, img_points, CAM_MATRIX, DIST_COEFFS
            )

            if success:
                cam_pos, cam_rot = get_camera_pose_in_world(rvec, tvec)
                
                # Optional: Draw axes on the frame for debugging
                cv2.drawFrameAxes(frame, CAM_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)
                
                print(f"[{camera_name}] Camera XYZ (World): {cam_pos.flatten()}")
                return cam_pos, cam_rot

    return None, None

def create_transform_matrix(tx, ty, tz, rz_deg):
    """Creates a 4x4 homogeneous transformation matrix for a Z-axis rotation."""
    rz_rad = math.radians(rz_deg)
    cos_a = math.cos(rz_rad)
    sin_a = math.sin(rz_rad)
    
    # Standard 4x4 Transformation Matrix (Rotation around Z + Translation)
    return np.array([
        [cos_a, -sin_a, 0, tx],
        [sin_a,  cos_a, 0, ty],
        [    0,      0, 1, tz],
        [    0,      0, 0,  1]
    ], dtype=np.float32)

# 1. Define Camera poses IN the Robot's local coordinate frame
# Cam 1: (0, -2in, 0), Rot Z: -90 deg
T_cam1_in_robot = create_transform_matrix(0.0, -2.0 * INCH_TO_M, 0.0, -90.0)

# Cam 2: (0, 2in, 0), Rot Z: 90 deg
T_cam2_in_robot = create_transform_matrix(0.0, 2.0 * INCH_TO_M, 0.0, 90.0)

def get_robot_pose(cam_pos_world, cam_rot_world, T_cam_in_robot):
    """Calculates Robot pose in World frame using matrix multiplication."""
    # Build 4x4 matrix for Camera in World
    T_cam_in_world = np.eye(4, dtype=np.float32)
    T_cam_in_world[:3, :3] = cam_rot_world
    T_cam_in_world[:3, 3] = cam_pos_world.flatten()
    
    # Calculate: T_robot_in_world = T_cam_in_world * (T_cam_in_robot)^-1
    T_robot_in_cam = np.linalg.inv(T_cam_in_robot)
    T_robot_in_world = np.dot(T_cam_in_world, T_robot_in_cam)
    
    # Return the translation (position) and rotation (orientation) of the robot
    return T_robot_in_world[:3, 3], T_robot_in_world[:3, :3]

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    # Initialize cameras (adjust indices as needed, e.g., 0 and 2)
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    # Set resolutions to be lightweight (e.g., 640x480)
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if ret1:
                pos1, rot1 = process_frame(frame1, "Cam 1 (Front)")
                if pos1 is not None:
                    robot_pos1, robot_rot1 = get_robot_pose(pos1, rot1, T_cam1_in_robot)
                    # Use numpy formatting to keep print statements readable
                    np.set_printoptions(precision=3, suppress=True)
                    print(f"[Cam 1] Robot Center (XYZ): {robot_pos1}")
                
            if ret2:
                pos2, rot2 = process_frame(frame2, "Cam 2 (Back)")
                if pos2 is not None:
                    robot_pos2, robot_rot2 = get_robot_pose(pos2, rot2, T_cam2_in_robot)
                    np.set_printoptions(precision=3, suppress=True)
                    print(f"[Cam 2] Robot Center (XYZ): {robot_pos2}")
            
            # --- COMBINING THE DATA ---
            # If both cameras see tags, you can average robot_pos1 and robot_pos2 
            # for a more stable position estimate. If only one sees a tag, use that one!
            
            #cv2.imshow("Cam 1", frame1)
            cv2.imshow("Cam 2", frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()