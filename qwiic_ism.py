#!/usr/bin/env python3
"""
imu_read.py
-----------
Reads all 9DOF data from the SparkFun ISM330DHCX + MMC5983MA breakout.
    - ISM330DHCX: accelerometer (g) + gyroscope (dps)
    - MMC5983MA:  magnetometer (gauss)

Requires: pip install sparkfun-qwiic-ism330dhcx sparkfun-qwiic-mmc5983ma
"""

import sys
import time
import qwiic_ism330dhcx
import qwiic_mmc5983ma

POLL_INTERVAL = 0.1  # seconds between readings (10 Hz)


def init_imu():
    imu = qwiic_ism330dhcx.QwiicISM330DHCX(address=0x6B)
    if not imu.is_connected():
        print("ISM330DHCX not found — check wiring and i2cdetect", file=sys.stderr)
        sys.exit(1)

    imu.begin()
    imu.device_reset()
    while imu.get_device_reset() == False:
        time.sleep(1)

    time.sleep(0.1)
    imu.set_device_config()
    imu.set_block_data_update()

    # Accelerometer: 104 Hz, ±4g, with low-pass filter
    imu.set_accel_data_rate(imu.kXlOdr104Hz)
    imu.set_accel_full_scale(imu.kXlFs4g)
    imu.set_accel_filter_lp2()
    imu.set_accel_slope_filter(imu.kLpOdrDiv100)

    # Gyroscope: 104 Hz, ±500 dps, with low-pass filter
    imu.set_gyro_data_rate(imu.kGyroOdr104Hz)
    imu.set_gyro_full_scale(imu.kGyroFs500dps)
    imu.set_gyro_filter_lp1()
    imu.set_gyro_lp1_bandwidth(imu.kBwMedium)

    print("ISM330DHCX initialised")
    return imu


def init_mag():
    mag = qwiic_mmc5983ma.QwiicMMC5983MA()
    if not mag.is_connected():
        print("MMC5983MA not found — check wiring and i2cdetect", file=sys.stderr)
        sys.exit(1)
    mag.begin()
    print("MMC5983MA initialised")
    return mag


def read_imu(imu):
    accel = imu.get_accel()
    gyro  = imu.get_gyro()
    return {
        "accel_x": accel.xData,
        "accel_y": accel.yData,
        "accel_z": accel.zData,
        "gyro_x":  gyro.xData,
        "gyro_y":  gyro.yData,
        "gyro_z":  gyro.zData,
    }


def read_mag(mag):
    return {
        "mag_x": mag.get_measurement_x(),
        "mag_y": mag.get_measurement_y(),
        "mag_z": mag.get_measurement_z(),
    }


def print_data(imu_data, mag_data):
    print(
        f"Accel (g)      "
        f"X: {imu_data['accel_x']:8.3f}  "
        f"Y: {imu_data['accel_y']:8.3f}  "
        f"Z: {imu_data['accel_z']:8.3f}"
    )
    print(
        f"Gyro  (dps)    "
        f"X: {imu_data['gyro_x']:8.3f}  "
        f"Y: {imu_data['gyro_y']:8.3f}  "
        f"Z: {imu_data['gyro_z']:8.3f}"
    )
    print(
        f"Mag   (gauss)  "
        f"X: {mag_data['mag_x']:8.3f}  "
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
            if imu.check_status():
                imu_data = read_imu(imu)
                mag_data  = read_mag(mag)
                print_data(imu_data, mag_data)
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")
