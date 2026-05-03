#!/usr/bin/env python3
"""
sensing_init.py
---------------
Initializes all sensors on the ManaTM boat:
    - IMU (ISM330DHCX)  via Qwiic/I2C  → accelerometer + gyroscope
    - Cameras (x2)      via USB         → ArUco pose estimation
    - Leak sensor       via GPIO4       → bilge leak detection
    - LiDAR (RPLidar C1) via USB       → /dev/ttyUSB0

Usage:
    from sensing_init import init_all
    imu, cap1, cap2, leak_sensor, lidar = init_all()
"""

import sys
import time
import qwiic_ism330dhcx
import cv2
from gpiozero import Button
from rplidarc1 import RPLidar

# ── Config ─────────────────────────────────────────────────────────────────────
IMU_ADDRESS     = 0x6b
LEAK_SENSOR_PIN = 4
LIDAR_PORT      = '/dev/ttyUSB0'
LIDAR_BAUD      = 460800
CAM1_INDEX      = 0
CAM2_INDEX      = 1
CAM_WIDTH       = 640
CAM_HEIGHT      = 480
CAM_FPS         = 30


def init_imu():
    """
    Initialize the ISM330DHCX IMU over Qwiic/I2C.
    Returns a ready-to-read QwiicISM330DHCX object.

    Read data with:
        if imu.check_status():
            accel = imu.get_accel()   # accel.xData, .yData, .zData  (mg)
            gyro  = imu.get_gyro()    # gyro.xData,  .yData, .zData  (mdps)
    """
    print("Initialising IMU...")
    myIsm = qwiic_ism330dhcx.QwiicISM330DHCX(address=IMU_ADDRESS)

    myIsm.begin()
    myIsm.device_reset()
    while myIsm.get_device_reset() == False:
        time.sleep(1)

    print("Reset complete, applying settings...")
    time.sleep(0.1)

    myIsm.set_device_config()
    myIsm.set_block_data_update()

    # Accelerometer: 104 Hz, ±4g, low-pass filter
    myIsm.set_accel_data_rate(myIsm.kXlOdr104Hz)
    myIsm.set_accel_full_scale(myIsm.kXlFs4g)
    myIsm.set_accel_filter_lp2()
    myIsm.set_accel_slope_filter(myIsm.kLpOdrDiv100)

    # Gyroscope: 104 Hz, ±500 dps, low-pass filter
    myIsm.set_gyro_data_rate(myIsm.kGyroOdr104Hz)
    myIsm.set_gyro_full_scale(myIsm.kGyroFs500dps)
    myIsm.set_gyro_filter_lp1()
    myIsm.set_gyro_lp1_bandwidth(myIsm.kBwMedium)

    print("IMU ready")
    return myIsm


def init_cameras():
    """
    Initialize both USB cameras for ArUco pose estimation.
    Returns (cap1, cap2) OpenCV VideoCapture objects.

    Read data with:
        ret, frame = cap1.read()
        if ret:
            pos, rot = process_frame(frame, "Cam 1")

    Remember to call cap1.release() and cap2.release() on shutdown.
    """
    print("Initialising cameras...")
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)

    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

    if not cap1.isOpened():
        print("Camera 1 not found — check USB connection", file=sys.stderr)
        sys.exit(1)
    if not cap2.isOpened():
        print("Camera 2 not found — check USB connection", file=sys.stderr)
        sys.exit(1)

    print("Cameras ready")
    return cap1, cap2


def init_leak_sensor():
    """
    Initialize the leak sensor on GPIO4.
    Returns a gpiozero Button object.

    Read with:
        if leak_sensor.is_pressed:   # True = leak detected
    """
    print("Initialising leak sensor...")
    leak = Button(LEAK_SENSOR_PIN, pull_up=False)
    print("Leak sensor ready")
    return leak


def init_lidar():
    """
    Initialize the RPLidar C1 on /dev/ttyUSB0.
    Returns a ready-to-use RPLidar object.

    Read with (async):
        async for point in lidar.iter_scans():
            angle    = int(point['a_deg']) % 360
            distance = point['d_mm']

    Remember to call lidar.reset() on shutdown.
    """
    print("Initialising LiDAR...")
    lidar = RPLidar(port=LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=0.2)
    print("LiDAR ready")
    return lidar


def init_all():
    """
    Convenience function — initializes all sensors at once.
    Returns (imu, cap1, cap2, leak_sensor, lidar)
    """
    imu         = init_imu()
    cap1, cap2  = init_cameras()
    leak_sensor = init_leak_sensor()
    lidar       = init_lidar()
    print("\nAll sensors initialised\n")
    return imu, cap1, cap2, leak_sensor, lidar


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    imu, cap1, cap2, leak, lidar = init_all()

    print("Testing IMU for 5 readings...")
    count = 0
    while count < 5:
        if imu.check_status():
            accel = imu.get_accel()
            gyro  = imu.get_gyro()
            print(
                f"Accel (mg)   "
                f"X: {accel.xData:8.3f}  "
                f"Y: {accel.yData:8.3f}  "
                f"Z: {accel.zData:8.3f}"
            )
            print(
                f"Gyro  (mdps) "
                f"X: {gyro.xData:8.3f}  "
                f"Y: {gyro.yData:8.3f}  "
                f"Z: {gyro.zData:8.3f}"
            )
            print("-" * 60)
            count += 1
        time.sleep(0.1)

    print(f"\nLeak sensor: {'LEAK DETECTED' if leak.is_pressed else 'dry'}")

    print("\nTesting cameras (single frame each)...")
    ret1, _ = cap1.read()
    ret2, _ = cap2.read()
    print(f"Camera 1: {'ok' if ret1 else 'FAILED'}")
    print(f"Camera 2: {'ok' if ret2 else 'FAILED'}")

    print("\nSensor check complete")
    cap1.release()
    cap2.release()
    lidar.reset()
