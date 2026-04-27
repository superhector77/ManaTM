# SPDX-FileCopyrightText: 2019 Dave Astels for Adafruit Industries
# SPDX-License-Identifier: MIT
"""
lidar_display.py
----------------
Reads RPLidar data and displays a live polar plot in a desktop window.
Requires: pip install adafruit-circuitpython-rplidar pygame
"""

import sys
from math import cos, sin, pi, floor
import pygame
from adafruit_rplidar import RPLidar

# ── Display settings ───────────────────────────────────────────────────────────
WINDOW_SIZE   = 600          # square window, pixels
CENTRE        = WINDOW_SIZE // 2
PLOT_RADIUS   = CENTRE - 20  # leave a small margin
BG_COLOR      = (0, 0, 0)
POINT_COLOR   = (255, 255, 255)
GRID_COLOR    = (30, 30, 30)
LABEL_COLOR   = (100, 100, 100)

# ── LiDAR settings ────────────────────────────────────────────────────────────
PORT_NAME    = '/dev/ttyUSB0'
MAX_DISTANCE = 5000          # mm — clamp anything beyond this

# ── Setup ──────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("RPLidar Live View")
font = pygame.font.SysFont(None, 18)

lidar = RPLidar(None, PORT_NAME)
scan_data = [0] * 360
max_distance = 0


def draw_grid():
    """Draw reference circles and crosshairs."""
    for r in [0.25, 0.5, 0.75, 1.0]:
        pygame.draw.circle(screen, GRID_COLOR, (CENTRE, CENTRE),
                           int(PLOT_RADIUS * r), 1)
    pygame.draw.line(screen, GRID_COLOR, (0, CENTRE), (WINDOW_SIZE, CENTRE), 1)
    pygame.draw.line(screen, GRID_COLOR, (CENTRE, 0), (CENTRE, WINDOW_SIZE), 1)

    # Distance labels on the rings
    for r, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
        text = font.render(label, True, LABEL_COLOR)
        screen.blit(text, (CENTRE + int(PLOT_RADIUS * r) + 2, CENTRE + 2))


def process_data(data):
    global max_distance
    screen.fill(BG_COLOR)
    draw_grid()

    for angle in range(360):
        distance = data[angle]
        if distance > 0:
            max_distance = max(min(MAX_DISTANCE, distance), max_distance)
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            px = CENTRE + int(x / max_distance * PLOT_RADIUS)
            py = CENTRE + int(y / max_distance * PLOT_RADIUS)
            pygame.draw.circle(screen, POINT_COLOR, (px, py), 2)

    # Max distance label
    dist_text = font.render(f"Max seen: {int(max_distance)}mm", True, LABEL_COLOR)
    screen.blit(dist_text, (10, 10))

    pygame.display.update()


try:
    print(lidar.info)
    for scan in lidar.iter_scans():
        # Handle quit event so the window close button works
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                raise KeyboardInterrupt

        for (_, angle, distance) in scan:
            scan_data[min(359, floor(angle))] = distance

        process_data(scan_data)

except KeyboardInterrupt:
    print("Stopping.")

finally:
    lidar.stop()
    lidar.disconnect()
    pygame.quit()
    sys.exit()
