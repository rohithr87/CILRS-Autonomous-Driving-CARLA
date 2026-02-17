#!/usr/bin/env python3
"""
Load CUSAT Campus map into CARLA and test drive
"""

import sys
import os
import time
import math
import random

CARLA_ROOT = "/home/rohith/carla_simulator"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla"))

import carla


def main():
    print("=" * 60)
    print("LOADING CUSAT CAMPUS INTO CARLA")
    print("=" * 60)

    # Connect
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    print("Connected to CARLA")

    # Read OpenDRIVE file
    xodr_path = os.path.expanduser('~/Downloads/cusat_campus.xodr')
    
    if not os.path.exists(xodr_path):
        print(f"ERROR: {xodr_path} not found!")
        return

    with open(xodr_path, 'r') as f:
        xodr_data = f.read()

    print(f"OpenDRIVE file: {len(xodr_data)} bytes")

    # Generate world from OpenDRIVE
    print("\nGenerating CUSAT campus world...")
    print("(This may take 10-30 seconds...)")

    try:
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=500.0,
            wall_height=0.0,
            additional_width=1.0
        )

        world = client.generate_opendrive_world(xodr_data, params)
        print("✅ CUSAT Campus world loaded!")

    except Exception as e:
        print(f"❌ Failed to load: {e}")
        print("\nTrying with different parameters...")
        
        try:
            params = carla.OpendriveGenerationParameters(
                vertex_distance=5.0,
                max_road_length=1000.0,
                wall_height=0.0,
                additional_width=2.0
            )
            world = client.generate_opendrive_world(xodr_data, params)
            print("✅ CUSAT Campus world loaded (attempt 2)!")
        except Exception as e2:
            print(f"❌ Failed again: {e2}")
            return

    time.sleep(3.0)
    world = client.get_world()

    # Get map info
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"\nMap name: {carla_map.name}")
    print(f"Spawn points: {len(spawn_points)}")

    if len(spawn_points) == 0:
        print("\n⚠ No spawn points generated!")
        print("Trying to get waypoints instead...")
        
        # Generate spawn points from waypoints
        waypoints = carla_map.generate_waypoints(20.0)
        print(f"Waypoints found: {len(waypoints)}")
        
        if waypoints:
            # Use first waypoint to place spectator
            wp = waypoints[0]
            spectator = world.get_spectator()
            spec_transform = carla.Transform(
                carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z + 50
                ),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(spec_transform)
            print(f"\n👁 Spectator placed at bird's eye view")
            print(f"   Location: ({wp.transform.location.x:.1f}, "
                  f"{wp.transform.location.y:.1f})")
            print("\n🔍 Look at the CARLA window — you should see CUSAT roads!")
        return

    # Place spectator for bird's eye view of campus
    # Calculate center of all spawn points
    avg_x = sum(sp.location.x for sp in spawn_points) / len(spawn_points)
    avg_y = sum(sp.location.y for sp in spawn_points) / len(spawn_points)

    spectator = world.get_spectator()

    # Bird's eye view first
    print("\n👁 Setting bird's eye view...")
    spec_transform = carla.Transform(
        carla.Location(x=avg_x, y=avg_y, z=200),
        carla.Rotation(pitch=-90)
    )
    spectator.set_transform(spec_transform)

    print(f"   Center: ({avg_x:.1f}, {avg_y:.1f})")
    print(f"\n🔍 Look at the CARLA window!")
    print(f"   You should see CUSAT campus road network from above!")

    time.sleep(5.0)

    # Spawn a test vehicle
    print("\n🚗 Spawning test vehicle...")
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]

    vehicle = None
    for sp in spawn_points[:20]:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            break

    if vehicle is None:
        print("Could not spawn vehicle!")
        return

    loc = vehicle.get_transform().location
    print(f"   Vehicle at: ({loc.x:.1f}, {loc.y:.1f})")

    # Move spectator behind vehicle
    time.sleep(2.0)
    car_t = vehicle.get_transform()
    yaw = math.radians(car_t.rotation.yaw)
    spectator.set_transform(carla.Transform(
        carla.Location(
            x=car_t.location.x - 8 * math.cos(yaw),
            y=car_t.location.y - 8 * math.sin(yaw),
            z=car_t.location.z + 5
        ),
        carla.Rotation(pitch=-20, yaw=car_t.rotation.yaw)
    ))

    print(f"\n" + "=" * 60)
    print(f"CUSAT CAMPUS MAP LOADED SUCCESSFULLY!")
    print(f"=" * 60)
    print(f"\n  Spawn points:  {len(spawn_points)}")
    print(f"  Test vehicle:  Spawned ✅")
    print(f"\n  Next step: Run autonomous_drive.py on this map")
    print(f"\n  Press Ctrl+C to exit...")

    # Print some spawn point locations for landmark mapping
    print(f"\n  Sample spawn locations (for landmark mapping):")
    for i, sp in enumerate(spawn_points[:15]):
        print(f"    Spawn {i:3d}: ({sp.location.x:8.1f}, {sp.location.y:8.1f})")

    # Keep running so we can see the map
    try:
        while True:
            # Slowly rotate bird's eye view
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nCleaning up...")
        vehicle.destroy()
        print("Done!")


if __name__ == '__main__':
    main()
