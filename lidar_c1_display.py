#!/usr/bin/env python3
"""
lidar_c1_display.py
-------------------
Live polar plot of RPLidar C1 data using pygame.
Requires: pip install rplidarc1 pyserial pygame
"""

import asyncio
import sys
from math import cos, sin, pi
import pygame
from rplidarc1 import RPLidar

# ── Settings ───────────────────────────────────────────────────────────────────
PORT_NAME    = '/dev/ttyUSB0'
WINDOW_SIZE  = 600
CENTRE       = WINDOW_SIZE // 2
PLOT_RADIUS  = CENTRE - 20
MAX_DISTANCE = 5000          # mm, clamp beyond this

BG_COLOR     = (0,   0,   0)
POINT_COLOR  = (255, 255, 255)
GRID_COLOR   = (30,  30,  30)
LABEL_COLOR  = (100, 100, 100)

# ── Shared scan buffer ─────────────────────────────────────────────────────────
scan_data    = [0] * 360
max_distance = 1             # avoid divide-by-zero at startup
running      = True


# ── pygame helpers ─────────────────────────────────────────────────────────────

def draw_grid(screen, font):
    for r in [0.25, 0.5, 0.75, 1.0]:
        pygame.draw.circle(screen, GRID_COLOR, (CENTRE, CENTRE),
                           int(PLOT_RADIUS * r), 1)
    pygame.draw.line(screen, GRID_COLOR, (0, CENTRE), (WINDOW_SIZE, CENTRE), 1)
    pygame.draw.line(screen, GRID_COLOR, (CENTRE, 0), (CENTRE, WINDOW_SIZE), 1)
    for r, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
        text = font.render(label, True, LABEL_COLOR)
        screen.blit(text, (CENTRE + int(PLOT_RADIUS * r) + 2, CENTRE + 2))


def render(screen, font):
    global max_distance
    screen.fill(BG_COLOR)
    draw_grid(screen, font)

    for angle in range(360):
        distance = scan_data[angle]
        if distance > 0:
            max_distance = max(min(MAX_DISTANCE, distance), max_distance)
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            px = CENTRE + int(x / max_distance * PLOT_RADIUS)
            py = CENTRE + int(y / max_distance * PLOT_RADIUS)
            pygame.draw.circle(screen, POINT_COLOR, (px, py), 2)

    info = font.render(f"Max seen: {int(max_distance)} mm", True, LABEL_COLOR)
    screen.blit(info, (10, 10))
    pygame.display.update()


# ── Lidar coroutine ────────────────────────────────────────────────────────────

async def read_lidar():
    global running
    lidar = RPLidar(port=PORT_NAME, baudrate=460800, timeout=0.2)
    try:
        async for point in lidar.iter_scans():
            if not running:
                break
            angle    = int(point['a_deg']) % 360
            distance = point['d_mm']
            if point['q'] > 0:              # skip zero-quality points
                scan_data[angle] = distance
    except Exception as e:
        print(f"Lidar error: {e}")
    finally:
        await lidar.stop()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    global running

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("RPLidar C1 Live View")
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    lidar_task = asyncio.create_task(read_lidar())

    print("Running — press Q or close window to quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        render(screen, font)
        clock.tick(30)          # cap at 30 fps

    lidar_task.cancel()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    asyncio.run(main())
