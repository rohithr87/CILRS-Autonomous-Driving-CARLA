#!/usr/bin/env python3
"""
Map CUSAT landmarks to CARLA spawn points
"""

import sys
import os
import math

CARLA_ROOT = "/home/rohith/carla_simulator"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla"))

import carla

# Reference point used in conversion
REF_LAT = 10.0475
REF_LON = 76.3250

def lat_lon_to_xy(lat, lon):
    R = 6371000.0
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    ref_lat_r = math.radians(REF_LAT)
    ref_lon_r = math.radians(REF_LON)
    x = R * (lon_r - ref_lon_r) * math.cos(ref_lat_r)
    y = R * (lat_r - ref_lat_r)
    return x, y

# CUSAT Landmark GPS coordinates (approximate from Google Maps)
CUSAT_LANDMARKS_GPS = {
    "CUSAT Main Gate":           (10.0465, 76.3180),
    "School of Engineering":     (10.0455, 76.3218),
    "SOE Central Avenue":        (10.0460, 76.3210),
    "Dept of Computer Science":  (10.0462, 76.3225),
    "Admin Building":            (10.0480, 76.3200),
    "Central Library":           (10.0472, 76.3215),
    "Seminar Complex":           (10.0485, 76.3210),
    "University Hostel":         (10.0490, 76.3230),
    "Sports Arena":              (10.0478, 76.3225),
    "Canteen":                   (10.0470, 76.3220),
    "Marine Sciences":           (10.0500, 76.3195),
    "Ship Technology":           (10.0495, 76.3185),
    "CUCEK":                     (10.0458, 76.3235),
    "Staff Quarters":            (10.0505, 76.3215),
    "Boat Jetty":                (10.0510, 76.3200),
    "Lakeside":                  (10.0498, 76.3240),
    "Vidya Nagar Junction":      (10.0445, 76.3200),
    "HMT Junction":              (10.0440, 76.3170),
    "University Road Start":     (10.0468, 76.3190),
    "Hostel Road":               (10.0492, 76.3220),
}

def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    print("=" * 70)
    print("CUSAT LANDMARK → SPAWN POINT MAPPING")
    print("=" * 70)

    # Convert landmarks to XY
    landmarks_xy = {}
    for name, (lat, lon) in CUSAT_LANDMARKS_GPS.items():
        x, y = lat_lon_to_xy(lat, lon)
        landmarks_xy[name] = (x, y)

    # Find nearest spawn point for each landmark
    print(f"\n{'Landmark':<30s} {'Spawn':>5s} {'Distance':>8s} {'XY Position':>20s}")
    print("-" * 70)

    landmark_mapping = {}

    for name, (lx, ly) in landmarks_xy.items():
        min_dist = float('inf')
        nearest_idx = 0
        nearest_loc = None

        for i, sp in enumerate(spawn_points):
            dx = sp.location.x - lx
            dy = sp.location.y - ly
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                nearest_loc = sp.location

        landmark_mapping[name] = nearest_idx

        print(f"  {name:<28s} #{nearest_idx:>3d}  {min_dist:>6.1f}m  "
              f"({nearest_loc.x:>7.1f}, {nearest_loc.y:>7.1f})")

    # Print Python dict for copy-paste
    print(f"\n{'=' * 70}")
    print("COPY THIS INTO YOUR autonomous_drive.py:")
    print(f"{'=' * 70}")
    print("\nCUSAT_LANDMARKS = {")
    for name, idx in landmark_mapping.items():
        print(f'    {idx}: "{name}",')
    print("}")

    # Also print good route pairs
    print(f"\n{'=' * 70}")
    print("SUGGESTED DEMO ROUTES:")
    print(f"{'=' * 70}")

    routes = [
        ("CUSAT Main Gate", "Admin Building"),
        ("School of Engineering", "Central Library"),
        ("CUSAT Main Gate", "University Hostel"),
        ("SOE Central Avenue", "Seminar Complex"),
        ("Canteen", "Sports Arena"),
        ("Admin Building", "Marine Sciences"),
    ]

    for start_name, end_name in routes:
        if start_name in landmark_mapping and end_name in landmark_mapping:
            s_idx = landmark_mapping[start_name]
            e_idx = landmark_mapping[end_name]
            s_sp = spawn_points[s_idx].location
            e_sp = spawn_points[e_idx].location
            dist = s_sp.distance(e_sp)
            print(f"\n  {start_name} → {end_name}")
            print(f"    --spawn {s_idx} --destination {e_idx}  (distance: {dist:.0f}m)")

    print(f"\n✅ Mapping complete!")
    print(f"   Total spawn points: {len(spawn_points)}")
    print(f"   Landmarks mapped: {len(landmark_mapping)}")


if __name__ == '__main__':
    main()
