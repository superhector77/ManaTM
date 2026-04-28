#!/usr/bin/env python3
"""
lidar_plot.py
-------------
Drop-in replacement for queue_printer that plots RPLidar C1 data live.
Run this instead of the original script — it imports everything from it.

Usage:
    python3 lidar_plot.py
"""

import asyncio
import time
import logging
import sys
from math import cos, sin, pi
import pygame
from rplidarc1 import RPLidar

# ── Display settings ───────────────────────────────────────────────────────────
WINDOW_SIZE  = 600
CENTRE       = WINDOW_SIZE // 2
PLOT_RADIUS  = CENTRE - 20
MAX_DISTANCE = 5000

BG_COLOR    = (0,   0,   0)
POINT_COLOR = (0, 255,   0)   # green points
GRID_COLOR  = (30,  30,  30)
LABEL_COLOR = (100, 100, 100)

# ── Shared state ───────────────────────────────────────────────────────────────
scan_data    = {}   # angle (int) -> distance (mm)
max_distance = 1


# ── pygame drawing ─────────────────────────────────────────────────────────────

def draw_grid(screen, font):
    for r in [0.25, 0.5, 0.75, 1.0]:
        pygame.draw.circle(screen, GRID_COLOR, (CENTRE, CENTRE),
                           int(PLOT_RADIUS * r), 1)
    pygame.draw.line(screen, GRID_COLOR, (0, CENTRE), (WINDOW_SIZE, CENTRE), 1)
    pygame.draw.line(screen, GRID_COLOR, (CENTRE, 0), (CENTRE, WINDOW_SIZE), 1)
    for r, label in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]:
        txt = font.render(label, True, LABEL_COLOR)
        screen.blit(txt, (CENTRE + int(PLOT_RADIUS * r) + 2, CENTRE + 2))


def render(screen, font):
    global max_distance
    screen.fill(BG_COLOR)
    draw_grid(screen, font)

    for angle, distance in scan_data.items():
        if distance > 0:
            max_distance = max(min(MAX_DISTANCE, distance), max_distance)
            radians = int(angle) * pi / 180.0
            x =  distance * cos(radians)
            y = -distance * sin(radians)   # flip y so 0° is up
            px = CENTRE + int(x / max_distance * PLOT_RADIUS)
            py = CENTRE + int(y / max_distance * PLOT_RADIUS)
            pygame.draw.circle(screen, POINT_COLOR, (px, py), 2)

    # HUD
    txt = font.render(f"Points: {len(scan_data)}   Max: {int(max_distance)} mm",
                      True, LABEL_COLOR)
    screen.blit(txt, (10, 10))
    pygame.display.update()


# ── Queue consumer (replaces queue_printer) ────────────────────────────────────

async def queue_plotter(q: asyncio.Queue, event: asyncio.Event,
                        screen, font, clock):
    """Read scan dicts from the queue and update the plot."""
    print("Plotter started")
    while not event.is_set():
        # Handle pygame events (window close / Q key)
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                event.set()
                return
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_q:
                event.set()
                return

        if q.qsize() < 1:
            await asyncio.sleep(0.05)
            continue

        point: dict = await q.get()

        # Each point dict has keys: a_deg, d_mm, q
        if point.get('q', 0) > 0:
            angle = int(point['a_deg']) % 360
            scan_data[angle] = point['d_mm']

        render(screen, font)
        clock.tick(30)


# ── Wait and stop (unchanged from original) ────────────────────────────────────

async def wait_and_stop(t, event: asyncio.Event):
    print(f"Scanning for {t} seconds...")
    await asyncio.sleep(t)
    print("Stopping scan")
    event.set()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(lidar: RPLidar):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("RPLidar C1 Live Plot")
    font  = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(wait_and_stop(30, lidar.stop_event))
        tg.create_task(queue_plotter(lidar.output_queue, lidar.stop_event,
                                     screen, font, clock))
        tg.create_task(lidar.simple_scan(make_return_dict=True))

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)   # suppress verbose lidar logs
    lidar = RPLidar("/dev/ttyUSB0", 460800)
    try:
        asyncio.run(main(lidar))
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(1)
        lidar.reset()
