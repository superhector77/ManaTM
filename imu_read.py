#!/usr/bin/env python3
"""
imu_read.py
-----------
Reads all 9DOF data from the SparkFun ISM330DHCX + MMC5983MA breakout.
    - ISM330DHCX: accelerometer + gyroscope (6DOF)
    - MMC5983MA:  magnetometer (3DOF)

Requires: pip install sparkfun-qwiic-ism330dhcx sparkfun-qwiic-mmc5983ma
"""

import time
import sys
import qwiic_ism330dhcx
import qwiic_mmc5983ma

POLL_INTERVAL = 0.1   # seconds between readings (10 Hz)


def init_imu():
    imu = qwiic_ism330dhcx.QwiicISM330DHCX()
    if not imu.is_connected():
        print("ISM330DHCX not found — check wiring and i2cdetect")
        sys.exit(1)
    imu.begin()
    print("ISM330DHCX connected")
    return imu


def init_mag():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        print("MMC5983MA not found — check wiring and i2cdetect")
        sys.exit(1)
    mag.begin()
    print("MMC5983MA connected")
    return mag


def read_imu(imu):
    imu.get_accel()
    imu.get_gyro()
    return {
        "accel_x": imu.sensor_data.accel.x,
        "accel_y": imu.sensor_data.accel.y,
        "accel_z": imu.sensor_data.accel.z,
        "gyro_x":  imu.sensor_data.gyro.x,
        "gyro_y":  imu.sensor_data.gyro.y,
        "gyro_z":  imu.sensor_data.gyro.z,
    }


def read_mag(mag):
    return {
        "mag_x": mag.get_measurement_x(),
        "mag_y": mag.get_measurement_y(),
        "mag_z": mag.get_measurement_z(),
    }


def print_data(imu_data, mag_data):
    print(
        f"Accel (g)    X: {imu_data['accel_x']:8.3f}  "
        f"Y: {imu_data['accel_y']:8.3f}  "
        f"Z: {imu_data['accel_z']:8.3f}"
    )
    print(
        f"Gyro (dps)   X: {imu_data['gyro_x']:8.3f}  "
        f"Y: {imu_data['gyro_y']:8.3f}  "
        f"Z: {imu_data['gyro_z']:8.3f}"
    )
    print(
        f"Mag (gauss)  X: {mag_data['mag_x']:8.3f}  "
        f"Y: {mag_data['mag_y']:8.3f}  "
        f"Z: {mag_data['mag_z']:8.3f}"
    )
    print("-" * 60)


if __name__ == "__main__":
    print("Initialising IMU...")
    imu = init_imu()
    mag = init_mag()
    print("Reading — Ctrl+C to stop\n")

    try:
        while True:
            imu_data = read_imu(imu)
            mag_data = read_mag(mag)
            print_data(imu_data, mag_data)
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")
