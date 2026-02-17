#!/usr/bin/env python3
# =============================================================================
# collect_data.py - CILRS Training Data Collection for CARLA 0.9.10
# =============================================================================
"""
Automated data collection using CARLA's autopilot.
Collects camera images + driving measurements for CILRS training.

Usage:
    python collect_data.py --session 1
    python collect_data.py --session 2 --vehicles 60 --weather cloudy
    python collect_data.py --session 3 --vehicles 30 --walkers 40 --weather sunset
    python collect_data.py --help

Author: Rohith
B.Tech CSE Major Project
"""

import os
import sys
import time
import random
import argparse
import csv
import signal
from datetime import datetime

import numpy as np
import cv2

# Add CARLA agents to path
sys.path.append('/home/rohith/carla_simulator/PythonAPI/carla')

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Data collection configuration."""
    
    # CARLA
    HOST = 'localhost'
    PORT = 2000
    TIMEOUT = 20.0
    TOWN = 'Town01'
    
    # Camera
    CAMERA_WIDTH = 800
    CAMERA_HEIGHT = 600
    CAMERA_FOV = 100
    CAMERA_X = 2.0      # Forward from vehicle center
    CAMERA_Y = 0.0      # Centered
    CAMERA_Z = 1.4      # Height (roughly eye level)
    
    # Recording
    FPS = 20
    JPEG_QUALITY = 95
    OUTPUT_BASE = '/home/rohith/carla_simulator/collected_data'
    
    # Autopilot
    TARGET_SPEED = 30.0  # km/h
    
    # Traffic Manager
    TM_PORT = 8000
    
    # Weather presets
    WEATHER = {
        'clear_noon': carla.WeatherParameters.ClearNoon,
        'clear_sunset': carla.WeatherParameters.ClearSunset,
        'cloudy': carla.WeatherParameters.CloudyNoon,
        'soft_rain': carla.WeatherParameters.SoftRainNoon,
        'clear_morning': carla.WeatherParameters(
            sun_altitude_angle=30.0,
            cloudiness=10.0,
            precipitation=0.0,
            wind_intensity=10.0,
        ),
    }
    
    # Predefined routes (start_index, end_index from spawn points)
    # These cover different areas of Town01 with many intersections
    ROUTES = {
        1:  (0, 100),
        2:  (50, 200),
        3:  (100, 10),
        4:  (150, 50),
        5:  (200, 80),
        6:  (30, 180),
        7:  (80, 220),
        8:  (120, 30),
        9:  (170, 60),
        10: (210, 130),
        11: (5, 250),
        12: (90, 15),
        13: (140, 200),
        14: (60, 150),
        15: (230, 40),
    }


class DataCollector:
    """Automated CARLA data collection system."""
    
    def __init__(self, args):
        self.args = args
        
        # State
        self.running = True
        self.frame_count = 0
        self.current_image = None
        
        # Actors to cleanup
        self.vehicle = None
        self.camera = None
        self.actors = []  # NPC vehicles and walkers
        self.walker_controllers = []
        
        # CARLA components
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.route_planner = None
        self.route = []
        self.route_index = 0
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = (
            f"session{args.session}"
            f"_{args.weather}"
            f"_v{args.vehicles}"
            f"_w{args.walkers}"
            f"_{timestamp}"
        )
        self.output_dir = os.path.join(Config.OUTPUT_BASE, session_name)
        self.image_dir = os.path.join(self.output_dir, 'images')
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C."""
        print("\n\n⚠️  Stopping collection gracefully...")
        self.running = False
    
    # -----------------------------------------------------------------
    # SETUP
    # -----------------------------------------------------------------
    
    def setup(self):
        """Initialize everything."""
        print("\n" + "=" * 65)
        print("🎬 CILRS DATA COLLECTION SYSTEM")
        print("=" * 65)
        print(f"   Session:      {self.args.session}")
        print(f"   Weather:      {self.args.weather}")
        print(f"   Vehicles:     {self.args.vehicles}")
        print(f"   Walkers:      {self.args.walkers}")
        print(f"   Duration:     {self.args.duration} minutes")
        print(f"   Routes:       {self.args.routes}")
        print(f"   Output:       {self.output_dir}")
        print("=" * 65)
        
        self._create_output_dirs()
        self._connect_carla()
        self._setup_weather()
        self._setup_traffic_manager()
        self._spawn_ego_vehicle()
        self._setup_camera()
        self._setup_route_planner()
        self._plan_route()
        self._spawn_npcs()
        self._setup_csv()
        
        # Let everything settle
        print("\n⏳ Warming up sensors...")
        for _ in range(40):
            self.world.tick()
        
        print("\n" + "=" * 65)
        print("✅ ALL SYSTEMS READY — STARTING COLLECTION")
        print("=" * 65 + "\n")
    
    def _create_output_dirs(self):
        """Create output directories."""
        os.makedirs(self.image_dir, exist_ok=True)
        print(f"\n📁 Output directory created")
    
    def _connect_carla(self):
        """Connect to CARLA and load Town01."""
        print(f"\n🔌 Connecting to CARLA...")
        
        self.client = carla.Client(Config.HOST, Config.PORT)
        self.client.set_timeout(Config.TIMEOUT)
        
        # Load correct map
        current_map = self.client.get_world().get_map().name
        if Config.TOWN not in current_map:
            print(f"   Loading {Config.TOWN}...")
            self.client.load_world(Config.TOWN)
            time.sleep(5.0)
        
        self.world = self.client.get_world()
        
        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / Config.FPS
        self.world.apply_settings(settings)
        
        print(f"   ✓ Connected: {self.world.get_map().name}")
        print(f"   ✓ Synchronous mode: {Config.FPS} FPS")
    
    def _setup_weather(self):
        """Set weather."""
        weather_name = self.args.weather
        weather = Config.WEATHER.get(weather_name, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)
        print(f"\n🌤️  Weather: {weather_name}")
    
    def _setup_traffic_manager(self):
        """Setup traffic manager for autopilot."""
        self.traffic_manager = self.client.get_trafficmanager(Config.TM_PORT)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        print(f"\n🚦 Traffic Manager ready (port {Config.TM_PORT})")
    
    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle (our recording car)."""
        print(f"\n🚗 Spawning ego vehicle...")
        
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Get preferred start point from route
        route_ids = self._get_route_ids()
        start_idx = Config.ROUTES[route_ids[0]][0]
        start_idx = min(start_idx, len(spawn_points) - 1)
        
        # Try preferred spawn point first, then others
        try_order = [start_idx]
        nearby = list(range(max(0, start_idx - 10), min(len(spawn_points), start_idx + 10)))
        random.shuffle(nearby)
        try_order.extend(nearby)
        
        # Add random points as last resort
        all_indices = list(range(len(spawn_points)))
        random.shuffle(all_indices)
        try_order.extend(all_indices)
        
        self.vehicle = None
        for idx in try_order:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[idx])
                break
            except RuntimeError:
                continue
        
        if self.vehicle is None:
            raise RuntimeError("Could not find any free spawn point!")
        
        # Enable autopilot
        self.vehicle.set_autopilot(True, Config.TM_PORT)
        
        # Set autopilot behavior
        self.traffic_manager.vehicle_percentage_speed_difference(
            self.vehicle, 
            100.0 * (1.0 - Config.TARGET_SPEED / 50.0)
        )
        # Respect traffic lights
        self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)
        
        loc = self.vehicle.get_transform().location
        print(f"   ✓ Spawned at: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
        print(f"   ✓ Autopilot enabled")

    def _respawn_ego_vehicle(self):
        """Respawn vehicle after destruction."""
        
        print(f"   Attempting respawn...")
        
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter('vehicle.tesla.model3')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        # Detach old camera
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except:
                pass
        
        # Spawn new vehicle
        self.vehicle = None
        for sp in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, sp)
                break
            except RuntimeError:
                continue
        
        if self.vehicle is None:
            raise RuntimeError("Could not respawn vehicle!")
        
        # Re-enable autopilot
        self.vehicle.set_autopilot(True, Config.TM_PORT)
        self.traffic_manager.vehicle_percentage_speed_difference(
            self.vehicle,
            100.0 * (1.0 - Config.TARGET_SPEED / 50.0)
        )
        self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)
        
        # Re-attach camera
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(Config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(Config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(Config.CAMERA_FOV))
        
        camera_transform = carla.Transform(
            carla.Location(x=Config.CAMERA_X, y=Config.CAMERA_Y, z=Config.CAMERA_Z)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self._on_camera_image)
        
        # Replan route from new position
        self._plan_new_route()
        
        loc = self.vehicle.get_transform().location
        print(f"   ✓ Respawned at: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
    
    def _setup_camera(self):
        """Attach RGB camera to vehicle."""
        print(f"\n📷 Setting up camera...")
        
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(Config.CAMERA_WIDTH))
        camera_bp.set_attribute('image_size_y', str(Config.CAMERA_HEIGHT))
        camera_bp.set_attribute('fov', str(Config.CAMERA_FOV))
        
        camera_transform = carla.Transform(
            carla.Location(x=Config.CAMERA_X, y=Config.CAMERA_Y, z=Config.CAMERA_Z)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(self._on_camera_image)
        
        print(f"   ✓ Camera: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}, FOV {Config.CAMERA_FOV}")
    
    def _on_camera_image(self, image):
        """Camera callback — store latest image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.current_image = array[:, :, :3]  # BGRA -> BGR
    
    def _setup_route_planner(self):
        """Initialize global route planner."""
        print(f"\n🗺️  Setting up route planner...")
        
        carla_map = self.world.get_map()
        dao = GlobalRoutePlannerDAO(carla_map, 2.0)
        self.route_planner = GlobalRoutePlanner(dao)
        self.route_planner.setup()
        
        print(f"   ✓ Route planner ready")
    
    def _plan_route(self):
        """Plan the driving route with waypoints and commands."""
        print(f"\n📍 Planning route...")
        
        spawn_points = self.world.get_map().get_spawn_points()
        route_ids = self._get_route_ids()
        
        self.route = []
        
        for route_id in route_ids:
            start_idx, end_idx = Config.ROUTES[route_id]
            start_idx = min(start_idx, len(spawn_points) - 1)
            end_idx = min(end_idx, len(spawn_points) - 1)
            
            start_loc = spawn_points[start_idx].location
            end_loc = spawn_points[end_idx].location
            
            try:
                segment = self.route_planner.trace_route(start_loc, end_loc)
                self.route.extend(segment)
                print(f"   ✓ Route {route_id}: {len(segment)} waypoints")
            except Exception as e:
                print(f"   ⚠ Route {route_id} failed: {e}")
       
        
        self.route_index = 0
        print(f"   ✓ Total route: {len(self.route)} waypoints")
        

    
    def _get_route_ids(self):
        """Parse route IDs from arguments."""
        route_str = self.args.routes
        
        if route_str == 'all':
            return list(Config.ROUTES.keys())
        
        ids = []
        for part in route_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                ids.extend(range(int(start), int(end) + 1))
            else:
                ids.append(int(part))
        
        # Validate
        valid_ids = [i for i in ids if i in Config.ROUTES]
        return valid_ids if valid_ids else [1, 2, 3]
    
    def _spawn_npcs(self):
        """Spawn NPC vehicles and pedestrians."""
        self._spawn_vehicles()
        self._spawn_walkers()
    
    def _spawn_vehicles(self):
        """Spawn NPC vehicles with autopilot."""
        num = self.args.vehicles
        if num <= 0:
            print(f"\n🚙 No NPC vehicles")
            return
        
        print(f"\n🚙 Spawning {num} NPC vehicles...")
        
        bp_library = self.world.get_blueprint_library()
        vehicle_bps = bp_library.filter('vehicle.*')
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        count = 0
        for i in range(min(num, len(spawn_points))):
            bp = random.choice(vehicle_bps)
            
            # Avoid bikes and motorcycles for stability
            if int(bp.get_attribute('number_of_wheels')) < 4:
                continue
            
            try:
                npc = self.world.spawn_actor(bp, spawn_points[i])
                npc.set_autopilot(True, Config.TM_PORT)
                self.actors.append(npc)
                count += 1
            except:
                pass
        
        print(f"   ✓ Spawned {count} vehicles")
    
    def _spawn_walkers(self):
        """Spawn pedestrian walkers."""
        num = self.args.walkers
        if num <= 0:
            print(f"\n🚶 No pedestrians")
            return
        
        print(f"\n🚶 Spawning {num} pedestrians...")
        
        bp_library = self.world.get_blueprint_library()
        walker_bps = bp_library.filter('walker.pedestrian.*')
        
        # Spawn walkers
        walkers = []
        batch = []
        
        for _ in range(num):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                bp = random.choice(walker_bps)
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                batch.append(carla.command.SpawnActor(bp, spawn_point))
        
        results = self.client.apply_batch_sync(batch, True)
        
        for result in results:
            if not result.error:
                walkers.append(result.actor_id)
        
        # Spawn walker controllers
        walker_controller_bp = bp_library.find('controller.ai.walker')
        batch = []
        
        for walker_id in walkers:
            batch.append(
                carla.command.SpawnActor(
                    walker_controller_bp,
                    carla.Transform(),
                    walker_id
                )
            )
        
        results = self.client.apply_batch_sync(batch, True)
        
        controllers = []
        for result in results:
            if not result.error:
                controllers.append(result.actor_id)
        
        # Let world tick once before starting controllers
        self.world.tick()
        
        # Start walking
        all_actors = self.world.get_actors(walkers + controllers)
        
        for actor in all_actors:
            if actor.type_id.startswith('controller'):
                actor.start()
                actor.go_to_location(
                    self.world.get_random_location_from_navigation()
                )
                actor.set_max_speed(1.0 + random.random() * 1.5)
        
        # Store for cleanup
        self.actors.extend(self.world.get_actors(walkers))
        self.walker_controllers = self.world.get_actors(controllers)
        
        print(f"   ✓ Spawned {len(walkers)} pedestrians")
    
    def _setup_csv(self):
        """Create measurements CSV file."""
        self.csv_path = os.path.join(self.output_dir, 'measurements.csv')
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame',
                'image_filename',
                'steer',
                'throttle',
                'brake',
                'speed_kmh',
                'speed_normalized',
                'high_level_command',
                'command_name',
                'position_x',
                'position_y',
                'position_z',
                'yaw',
                'timestamp'
            ])
        
        print(f"\n📝 CSV ready: {self.csv_path}")
    
    # -----------------------------------------------------------------
    # NAVIGATION COMMAND
    # -----------------------------------------------------------------
    
    def _get_navigation_command(self, vehicle_location):
        """
        Get current high-level navigation command.
        
        Returns:
            tuple: (command_index, command_name)
                   0=Follow Lane, 1=Left, 2=Right, 3=Straight
        """
        if not self.route or self.route_index >= len(self.route):
            return 0, 'LANEFOLLOW'
        
        # Find closest waypoint in route
        min_dist = float('inf')
        closest_idx = self.route_index
        
        # Search in a window around current index
        search_start = max(0, self.route_index - 5)
        search_end = min(len(self.route), self.route_index + 30)
        
        for i in range(search_start, search_end):
            wp = self.route[i][0]
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Advance route index
        if closest_idx > self.route_index:
            self.route_index = closest_idx
        
        # Look ahead a few waypoints for upcoming command
        look_ahead = min(self.route_index + 5, len(self.route) - 1)
        
        for i in range(self.route_index, look_ahead + 1):
            _, road_option = self.route[i]
            option_str = str(road_option).upper()
            
            if 'LEFT' in option_str:
                return 1, 'LEFT'
            elif 'RIGHT' in option_str:
                return 2, 'RIGHT'
            elif 'STRAIGHT' in option_str:
                return 3, 'STRAIGHT'
        
        return 0, 'LANEFOLLOW'
    
    # -----------------------------------------------------------------
    # MAIN COLLECTION LOOP
    # -----------------------------------------------------------------
    
    def collect(self):
        """Main data collection loop."""
        self.setup()
        
        duration_seconds = self.args.duration * 60
        start_time = time.time()
        
        # Statistics
        command_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        command_names = {0: 'LANEFOLLOW', 1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT'}
        last_print_time = start_time
        
        print(f"\n🔴 RECORDING... (Press Ctrl+C to stop)\n")
        
        try:
            while self.running:
                # Tick simulation
                self.world.tick()
                
                # Check duration
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    print(f"\n⏰ Duration reached ({self.args.duration} min)")
                    break
                
                # Skip if no image yet
                if self.current_image is None:
                    continue
                
                # Get vehicle state (handle destroyed vehicle)
                try:
                    transform = self.vehicle.get_transform()
                    location = transform.location
                    rotation = transform.rotation
                    velocity = self.vehicle.get_velocity()
                    control = self.vehicle.get_control()
                except RuntimeError:
                    print(f"\n⚠️  Vehicle destroyed! Respawning...")
                    try:
                        self._respawn_ego_vehicle()
                        # Wait for new vehicle to settle
                        for _ in range(20):
                            self.world.tick()
                        continue
                    except Exception as e:
                        print(f"   ❌ Respawn failed: {e}")
                        print(f"   Stopping collection.")
                        break
                
                # Calculate speed
                speed_kmh = 3.6 * np.sqrt(
                    velocity.x**2 + velocity.y**2 + velocity.z**2
                )
                speed_normalized = min(speed_kmh / 90.0, 1.0)
                
                # Skip if vehicle is not moving and no brake (likely stuck)
                if speed_kmh < 0.1 and control.brake < 0.1 and self.frame_count > 100:
                    continue
                
                # Get navigation command
                cmd_index, cmd_name = self._get_navigation_command(location)
                
                # Save image
                image_filename = f"frame_{self.frame_count:08d}.jpg"
                image_path = os.path.join(self.image_dir, image_filename)
                
                # Copy image to avoid race condition
                image = self.current_image.copy()
                
                # Save as JPEG
                cv2.imwrite(
                    image_path,
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
                )
                
                # Save measurements
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.frame_count,
                        image_filename,
                        round(control.steer, 6),
                        round(control.throttle, 6),
                        round(control.brake, 6),
                        round(speed_kmh, 2),
                        round(speed_normalized, 6),
                        cmd_index,
                        cmd_name,
                        round(location.x, 2),
                        round(location.y, 2),
                        round(location.z, 2),
                        round(rotation.yaw, 2),
                        round(elapsed, 3)
                    ])
                
                # Update statistics
                command_counts[cmd_index] += 1
                self.frame_count += 1
                
                # Print progress every 10 seconds
                if time.time() - last_print_time >= 10.0:
                    remaining = duration_seconds - elapsed
                    progress = (elapsed / duration_seconds) * 100
                    
                    print(
                        f"   📊 Frame {self.frame_count:7d} | "
                        f"Time {elapsed/60:5.1f}/{self.args.duration} min | "
                        f"Progress {progress:5.1f}% | "
                        f"Speed {speed_kmh:5.1f} km/h | "
                        f"Cmd: {cmd_name:10s} | "
                        f"L:{command_counts[0]} "
                        f"⬅:{command_counts[1]} "
                        f"➡:{command_counts[2]} "
                        f"⬆:{command_counts[3]}"
                    )
                    last_print_time = time.time()
                
                # Check if route is exhausted
                if self.route_index >= len(self.route) - 10:
                    print(f"\n🔄 Route completed, replanning...")
                    self._plan_new_route()
        
        except Exception as e:
            print(f"\n❌ Error during collection: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._save_summary(command_counts, command_names, time.time() - start_time)
            self.cleanup()
    
    def _plan_new_route(self):
        """Plan a new random route when current is exhausted."""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_loc = self.vehicle.get_transform().location
        
        # Pick random destination
        dest = random.choice(spawn_points).location
        
        try:
            new_segment = self.route_planner.trace_route(vehicle_loc, dest)
            self.route = new_segment
            self.route_index = 0
            print(f"   ✓ New route: {len(self.route)} waypoints")
        except Exception as e:
            print(f"   ⚠ Replanning failed: {e}")
    
    # -----------------------------------------------------------------
    # SUMMARY AND CLEANUP
    # -----------------------------------------------------------------
    
    def _save_summary(self, command_counts, command_names, duration):
        """Save collection summary."""
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        
        total_frames = sum(command_counts.values())
        
        lines = [
            "=" * 60,
            "DATA COLLECTION SUMMARY",
            "=" * 60,
            f"Session:          {self.args.session}",
            f"Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration:         {duration/60:.1f} minutes",
            f"Total Frames:     {total_frames}",
            f"FPS (recorded):   {total_frames/duration:.1f}" if duration > 0 else "FPS: N/A",
            f"Weather:          {self.args.weather}",
            f"NPC Vehicles:     {self.args.vehicles}",
            f"Pedestrians:      {self.args.walkers}",
            f"",
            "COMMAND DISTRIBUTION:",
            "-" * 40,
        ]
        
        for cmd_idx in sorted(command_counts.keys()):
            count = command_counts[cmd_idx]
            name = command_names[cmd_idx]
            pct = (count / total_frames * 100) if total_frames > 0 else 0
            lines.append(f"   {name:12s}: {count:8d} ({pct:5.1f}%)")
        
        lines.extend([
            "-" * 40,
            f"   TOTAL:       {total_frames:8d}",
            "",
            f"Output:          {self.output_dir}",
            f"Images:          {self.image_dir}",
            f"Measurements:    {self.csv_path}",
            "=" * 60,
        ])
        
        summary_text = "\n".join(lines)
        
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"\n{summary_text}")
    
    def cleanup(self):
        """Clean up all CARLA actors and settings."""
        print("\n🧹 Cleaning up...")
        
        # Reset synchronous mode FIRST
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except:
                pass
        
        # Small delay to let CARLA process
        time.sleep(1.0)
        
        # Stop camera
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except:
            pass
        self.camera = None
        
        # Disable autopilot before destroying
        try:
            if self.vehicle is not None:
                self.vehicle.set_autopilot(False)
        except:
            pass
        
        # Collect all actor IDs for batch destroy
        destroy_ids = []
        
        for controller in self.walker_controllers:
            try:
                controller.stop()
                destroy_ids.append(controller.id)
            except:
                pass
        
        for actor in self.actors:
            try:
                destroy_ids.append(actor.id)
            except:
                pass
        
        if self.vehicle is not None:
            try:
                destroy_ids.append(self.vehicle.id)
            except:
                pass
        
        # Batch destroy using commands
        if destroy_ids:
            try:
                batch = [carla.command.DestroyActor(x) for x in destroy_ids]
                self.client.apply_batch_sync(batch)
            except:
                pass
        
        self.walker_controllers = []
        self.actors = []
        self.vehicle = None
        
        print("   ✓ Cleanup complete!")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CILRS Training Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SESSION PRESETS (recommended collection plan):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Session 1 (Basic):
  python collect_data.py --session 1 --weather clear_noon --vehicles 20 --walkers 20 --duration 40 --routes 1,2,3,4,5

Session 2 (Heavy Traffic):
  python collect_data.py --session 2 --weather clear_noon --vehicles 60 --walkers 40 --duration 40 --routes 6,7,8,9,10

Session 3 (Sunset):
  python collect_data.py --session 3 --weather clear_sunset --vehicles 30 --walkers 20 --duration 30 --routes 1,3,5,7,9

Session 4 (Cloudy):
  python collect_data.py --session 4 --weather cloudy --vehicles 40 --walkers 30 --duration 30 --routes 2,4,6,8,10

Session 5 (Extra Turns):
  python collect_data.py --session 5 --weather clear_noon --vehicles 20 --walkers 10 --duration 20 --routes all
        """
    )
    
    parser.add_argument(
        '--session', type=int, default=1,
        help='Session number for identification (default: 1)'
    )
    parser.add_argument(
        '--weather', type=str, default='clear_noon',
        choices=['clear_noon', 'clear_sunset', 'cloudy', 'soft_rain', 'clear_morning'],
        help='Weather preset (default: clear_noon)'
    )
    parser.add_argument(
        '--vehicles', type=int, default=20,
        help='Number of NPC vehicles (default: 20)'
    )
    parser.add_argument(
        '--walkers', type=int, default=20,
        help='Number of pedestrians (default: 20)'
    )
    parser.add_argument(
        '--duration', type=int, default=30,
        help='Recording duration in minutes (default: 30)'
    )
    parser.add_argument(
        '--routes', type=str, default='1,2,3',
        help='Route IDs: "1,2,3" or "1-5" or "all" (default: 1,2,3)'
    )
    
    args = parser.parse_args()
    
    # Print session info
    print("\n" + "━" * 65)
    print("   🎬 CILRS DATA COLLECTION")
    print("   B.Tech CSE Major Project")
    print("━" * 65)
    
    collector = DataCollector(args)
    collector.collect()
    
    print("\n✅ Done! Data saved to:")
    print(f"   {collector.output_dir}\n")


if __name__ == '__main__':
    main()
