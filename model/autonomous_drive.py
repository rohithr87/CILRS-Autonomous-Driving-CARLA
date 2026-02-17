#!/usr/bin/env python3
"""
CILRS Autonomous Driving in CARLA 0.9.10 - V4.0 FINAL
Features: Traffic Lights + Dashboard HUD + Evaluation Metrics
          + Traffic Safety + Smooth Physics
"""

import sys
import os
import time
import math
import random
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque

# =================================================================
# CARLA Path Setup
# =================================================================
CARLA_ROOT = "/home/rohith/carla_simulator"
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"))
sys.path.insert(0, os.path.join(CARLA_ROOT, "PythonAPI/carla"))

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# =================================================================
# Numpy fix
# =================================================================
class _NumpyCoreModule:
    def __init__(self):
        import numpy.core.multiarray as ma
        import numpy.core.numeric as nu
        self.multiarray = ma
        self.numeric = nu

sys.modules['numpy._core'] = _NumpyCoreModule()
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric


# =================================================================
# EVALUATION METRICS
# =================================================================
class DrivingMetrics:
    """Track comprehensive driving evaluation metrics"""

    def __init__(self):
        self.total_distance = 0.0
        self.total_time = 0.0
        self.collisions = 0
        self.collision_types = {}
        self.red_light_violations = 0
        self.red_light_stops = 0
        self.off_road_frames = 0
        self.total_frames = 0
        self.routes_completed = 0
        self.routes_attempted = 0
        self.speeds = []
        self.steer_changes = []
        self.last_steer = 0.0
        self.last_position = None
        self.brake_events = 0
        self.obstacle_brakes = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self, speed_kmh, steer, on_road, dt):
        self.total_time += dt
        self.total_distance += speed_kmh * dt / 3.6
        self.speeds.append(speed_kmh)
        self.total_frames += 1

        # Steering smoothness (jerk)
        jerk = abs(steer - self.last_steer)
        self.steer_changes.append(jerk)
        self.last_steer = steer

        if not on_road:
            self.off_road_frames += 1

    def add_collision(self, actor_type):
        self.collisions += 1
        self.collision_types[actor_type] = self.collision_types.get(actor_type, 0) + 1

    def safety_score(self):
        score = 100.0
        score -= self.collisions * 15
        score -= self.red_light_violations * 10
        if self.total_frames > 0:
            off_road_pct = self.off_road_frames / self.total_frames
            score -= off_road_pct * 40
        return max(0.0, min(100.0, score))

    def comfort_score(self):
        if not self.steer_changes:
            return 100.0
        avg_jerk = np.mean(self.steer_changes)
        # Lower jerk = higher comfort (0.0 jerk = 100, 0.1 jerk = 0)
        return max(0.0, min(100.0, 100.0 - avg_jerk * 1000))

    def route_completion_rate(self):
        if self.routes_attempted == 0:
            return 0.0
        return self.routes_completed / self.routes_attempted * 100

    def print_report(self):
        print("\n" + "=" * 60)
        print("📊 EVALUATION REPORT")
        print("=" * 60)

        print(f"\n  {'─' * 50}")
        print(f"  DRIVING STATISTICS")
        print(f"  {'─' * 50}")
        print(f"  Total distance:      {self.total_distance:.0f} m ({self.total_distance/1000:.2f} km)")
        print(f"  Total time:          {self.total_time:.1f} s ({self.total_time/60:.1f} min)")
        print(f"  Average speed:       {np.mean(self.speeds):.1f} km/h")
        print(f"  Max speed:           {np.max(self.speeds):.1f} km/h")
        print(f"  Total frames:        {self.total_frames}")

        print(f"\n  {'─' * 50}")
        print(f"  ROUTE PERFORMANCE")
        print(f"  {'─' * 50}")
        print(f"  Routes attempted:    {self.routes_attempted}")
        print(f"  Routes completed:    {self.routes_completed}")
        print(f"  Completion rate:     {self.route_completion_rate():.1f}%")

        print(f"\n  {'─' * 50}")
        print(f"  SAFETY")
        print(f"  {'─' * 50}")
        print(f"  Total collisions:    {self.collisions}")
        if self.collision_types:
            for ctype, count in sorted(self.collision_types.items(),
                                       key=lambda x: -x[1]):
                print(f"    - {ctype}: {count}")
        print(f"  Red light violations:{self.red_light_violations}")
        print(f"  Red light stops:     {self.red_light_stops}")
        off_road_pct = (self.off_road_frames / max(self.total_frames, 1)) * 100
        print(f"  Off-road:            {off_road_pct:.1f}%")
        print(f"  Obstacle brakes:     {self.obstacle_brakes}")

        print(f"\n  {'─' * 50}")
        print(f"  SCORES")
        print(f"  {'─' * 50}")
        safety = self.safety_score()
        comfort = self.comfort_score()
        overall = (safety * 0.6 + comfort * 0.3 +
                   self.route_completion_rate() * 0.1)
        print(f"  Safety score:        {safety:.1f} / 100")
        print(f"  Comfort score:       {comfort:.1f} / 100")
        print(f"  Overall score:       {overall:.1f} / 100")

        # Grade
        if overall >= 90:
            grade = "A+ (Excellent)"
        elif overall >= 80:
            grade = "A  (Very Good)"
        elif overall >= 70:
            grade = "B+ (Good)"
        elif overall >= 60:
            grade = "B  (Satisfactory)"
        else:
            grade = "C  (Needs Improvement)"
        print(f"  Grade:               {grade}")
        print(f"\n  {'=' * 50}")


# =================================================================
# DASHBOARD HUD
# =================================================================
class DashboardHUD:
    """Live camera feed with Tesla-style HUD overlay"""

    def __init__(self):
        self.window_name = "CILRS Autonomous Driving"
        self.initialized = False

    def initialize(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)
        self.initialized = True

    def update(self, image, speed_kmh, cmd_name, steer, gas, brake,
               obs_dist, traffic_light, remaining, status, metrics):
        """Render HUD overlay on camera image"""
        if not self.initialized:
            self.initialize()

        # Resize for display
        display = cv2.resize(image, (960, 540))
        h, w = display.shape[:2]

        # ─── Bottom info bar (semi-transparent) ───
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 140), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.65, display, 0.35, 0)

        # ─── Top bar ───
        overlay2 = display.copy()
        cv2.rectangle(overlay2, (0, 0), (w, 50), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay2, 0.5, display, 0.5, 0)

        # Title
        cv2.putText(display, "CILRS AUTONOMOUS DRIVING", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Elapsed time
        elapsed_str = f"{metrics.total_time:.0f}s"
        cv2.putText(display, elapsed_str, (w - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ─── Traffic Light Indicator (top right) ───
        light_x = w - 55
        light_y = 80
        # Background circle
        cv2.circle(display, (light_x, light_y), 25, (40, 40, 40), -1)
        cv2.circle(display, (light_x, light_y), 25, (100, 100, 100), 2)
        if traffic_light == "RED":
            cv2.circle(display, (light_x, light_y), 20, (0, 0, 255), -1)
            cv2.putText(display, "STOP", (light_x - 22, light_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif traffic_light == "YELLOW":
            cv2.circle(display, (light_x, light_y), 20, (0, 220, 255), -1)
        else:
            cv2.circle(display, (light_x, light_y), 20, (0, 200, 0), -1)

        # ─── Speed (large, bottom left) ───
        speed_text = f"{speed_kmh:.0f}"
        cv2.putText(display, speed_text, (30, h - 55),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 255, 0), 3)
        cv2.putText(display, "km/h", (30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

        # ─── Command (colored badge) ───
        cmd_colors = {
            "FOLLOW": (0, 200, 0),
            "LEFT": (0, 255, 255),
            "RIGHT": (255, 200, 0),
            "STRAIGHT": (255, 165, 0)
        }
        cmd_color = cmd_colors.get(cmd_name, (255, 255, 255))

        # Command badge background
        cmd_x = 200
        badge_w = len(cmd_name) * 18 + 20
        cv2.rectangle(display, (cmd_x, h - 100), (cmd_x + badge_w, h - 70),
                      cmd_color, -1)
        cv2.putText(display, cmd_name, (cmd_x + 10, h - 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # ─── Steering Indicator (visual bar) ───
        steer_center_x = w // 2
        steer_y = h - 45
        bar_width = 200

        # Background bar
        cv2.rectangle(display,
                      (steer_center_x - bar_width // 2, steer_y - 8),
                      (steer_center_x + bar_width // 2, steer_y + 8),
                      (60, 60, 60), -1)

        # Center line
        cv2.line(display, (steer_center_x, steer_y - 12),
                 (steer_center_x, steer_y + 12), (255, 255, 255), 2)

        # Steering position
        steer_px = int(steer * bar_width / 2)
        steer_color = (0, 200, 255) if abs(steer) < 0.1 else (0, 140, 255)
        if steer_px != 0:
            x1 = steer_center_x
            x2 = steer_center_x + steer_px
            if x1 > x2:
                x1, x2 = x2, x1
            cv2.rectangle(display, (x1, steer_y - 6), (x2, steer_y + 6),
                          steer_color, -1)

        # Steering value text
        cv2.putText(display, f"STR: {steer:+.3f}",
                    (steer_center_x - 50, steer_y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ─── Throttle / Brake bars (right side) ───
        bar_x = w - 200
        # Throttle
        cv2.putText(display, "GAS", (bar_x, h - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        gas_w = int(gas * 120)
        cv2.rectangle(display, (bar_x + 40, h - 115),
                      (bar_x + 40 + gas_w, h - 100), (0, 200, 0), -1)
        cv2.rectangle(display, (bar_x + 40, h - 115),
                      (bar_x + 160, h - 100), (80, 80, 80), 1)

        # Brake
        cv2.putText(display, "BRK", (bar_x, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
        brk_w = int(brake * 120)
        cv2.rectangle(display, (bar_x + 40, h - 90),
                      (bar_x + 40 + brk_w, h - 75), (0, 0, 220), -1)
        cv2.rectangle(display, (bar_x + 40, h - 90),
                      (bar_x + 160, h - 75), (80, 80, 80), 1)

        # ─── Obstacle Warning ───
        if obs_dist < 18.0:
            if obs_dist < 6.0:
                warn_color = (0, 0, 255)
                warn_text = f"!! OBSTACLE {obs_dist:.1f}m !!"
            elif obs_dist < 12.0:
                warn_color = (0, 140, 255)
                warn_text = f"! CAUTION {obs_dist:.1f}m"
            else:
                warn_color = (0, 200, 255)
                warn_text = f"Ahead {obs_dist:.1f}m"

            text_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(display, warn_text, (text_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, warn_color, 2)

        # ─── Distance to destination ───
        cv2.putText(display, f"DEST: {remaining:.0f}m",
                    (bar_x, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ─── Status ───
        status_color = (0, 255, 0) if status == "OK" else (0, 200, 255)
        cv2.putText(display, status, (bar_x, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # ─── Metrics mini display (top left) ───
        safety = metrics.safety_score()
        cv2.putText(display, f"Safety: {safety:.0f}%", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if safety > 80 else (0, 165, 255), 1)
        cv2.putText(display, f"Dist: {metrics.total_distance:.0f}m",
                    (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(display,
                    f"Collisions: {metrics.collisions}", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if metrics.collisions == 0 else (0, 0, 255), 1)

        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        return key != 27  # Return False if ESC pressed

    def close(self):
        cv2.destroyAllWindows()


# =================================================================
# CILRS Model
# =================================================================
class CILRS(nn.Module):
    def __init__(self, num_commands=4, dropout=0.0):
        super(CILRS, self).__init__()
        self.num_commands = num_commands
        resnet = models.resnet34(pretrained=False)
        self.visual_encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool, nn.Flatten(),
        )
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        combined_dim = 640
        self.control_branches = nn.ModuleList()
        for _ in range(num_commands):
            self.control_branches.append(nn.Sequential(
                nn.Linear(combined_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(256, 3),
            ))
        self.speed_predictor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, image, speed, command):
        visual_features = self.visual_encoder(image)
        speed_features = self.speed_encoder(speed.unsqueeze(1))
        combined = torch.cat([visual_features, speed_features], dim=1)
        pred_speed = self.speed_predictor(visual_features).squeeze(1)
        batch_size = image.size(0)
        all_outputs = torch.stack(
            [branch(combined) for branch in self.control_branches], dim=0)
        command_idx = command.unsqueeze(0).unsqueeze(2).expand(1, batch_size, 3)
        controls = all_outputs.gather(0, command_idx).squeeze(0)
        return controls, pred_speed


# =================================================================
# Route Planner
# =================================================================
class RoutePlanner:
    ROAD_OPTION_TO_CMD = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 0, 6: 0}
    CMD_NAMES = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}

    def __init__(self, carla_map, sampling_resolution=2.0):
        dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution)
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()
        self.route = []
        self.current_idx = 0

    def set_route(self, start_location, end_location):
        self.route = self.planner.trace_route(start_location, end_location)
        self.current_idx = 0
        print(f"  Route planned: {len(self.route)} waypoints")
        return len(self.route) > 0

    def get_command(self, vehicle_location):
        if not self.route:
            return 0, "FOLLOW"
        min_dist = float('inf')
        closest_idx = self.current_idx
        search_start = max(0, self.current_idx - 5)
        search_end = min(len(self.route), self.current_idx + 50)
        for i in range(search_start, search_end):
            dist = vehicle_location.distance(self.route[i][0].transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        self.current_idx = closest_idx
        for look_offset in [3, 5, 8, 12]:
            look_idx = min(self.current_idx + look_offset, len(self.route) - 1)
            road_option = self.route[look_idx][1]
            cmd_val = road_option.value if hasattr(road_option, 'value') else road_option
            cmd_idx = self.ROAD_OPTION_TO_CMD.get(cmd_val, 0)
            if cmd_idx != 0:
                return cmd_idx, self.CMD_NAMES[cmd_idx]
        look_idx = min(self.current_idx + 8, len(self.route) - 1)
        road_option = self.route[look_idx][1]
        cmd_val = road_option.value if hasattr(road_option, 'value') else road_option
        cmd_idx = self.ROAD_OPTION_TO_CMD.get(cmd_val, 0)
        return cmd_idx, self.CMD_NAMES[cmd_idx]

    def get_next_waypoint_direction(self, vehicle_transform):
        if not self.route or self.current_idx >= len(self.route) - 1:
            return 0.0
        look_idx = min(self.current_idx + 5, len(self.route) - 1)
        target_loc = self.route[look_idx][0].transform.location
        vehicle_loc = vehicle_transform.location
        yaw = math.radians(vehicle_transform.rotation.yaw)
        fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            return 0.0
        cross = fwd_x * dy - fwd_y * dx
        return float(np.clip(cross / max(dist, 1.0), -1.0, 1.0))

    def is_route_complete(self, vehicle_location, threshold=10.0):
        if not self.route:
            return True
        return vehicle_location.distance(
            self.route[-1][0].transform.location) < threshold

    def distance_remaining(self, vehicle_location):
        if not self.route:
            return 0
        return vehicle_location.distance(
            self.route[-1][0].transform.location)


# =================================================================
# Autonomous Driver V4.0
# =================================================================
class AutonomousDriver:
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    IMG_WIDTH = 200
    IMG_HEIGHT = 88
    SPEED_NORM_FACTOR = 90.0

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Device: {self.device}")

        self.model = CILRS(num_commands=4, dropout=0.0).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Model loaded: epoch {checkpoint['epoch']}, "
              f"val_loss {checkpoint['val_loss']:.4f}")

        self.transform = transforms.Compose([
            transforms.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD)
        ])

        self.steer_history = deque(maxlen=5)
        self.throttle_history = deque(maxlen=5)

        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.route_planner = None

        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []

        self.latest_image = None
        self.image_lock = threading.Lock()
        self.had_collision = False
        self.collision_time = 0
        self.last_collision_print = 0
        self.collision_count = 0
        self.collision_cooldown = {}  # actor_type -> last_collision_time
        self.consecutive_collision_recoveries = 0
        self.running = False

        self.position_history = deque(maxlen=100)
        self.stopped_start_time = None
        self.waiting_for_traffic = False
        self.traffic_wait_start = None
        self.waiting_for_red = False
        self.off_road_frames_consecutive = 0
        self.red_light_clear_time = 0  # time when red light ended
        self.cached_nearby_actors = []
        self.actor_cache_frame = 0


        # Overtake state tracking
        self.overtake_state = "NONE"
        self.overtake_start_time = None
        self.obstacle_wait_start = None
        self.obstacle_wait_threshold = 4.0

        # New: Metrics and HUD
        self.metrics = DrivingMetrics()
        self.hud = DashboardHUD()

    def connect(self, host='localhost', port=2000, timeout=10.0, use_custom_map=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        current_map = self.world.get_map().name
        print(f"Current map: {current_map}")
        
        # Only load Town01 if NOT using a custom map
        if not use_custom_map:
            if "Town01" not in current_map:
                print("Loading Town01...")
                self.client.load_world('Town01')
                time.sleep(5.0)
                self.world = self.client.get_world()
        else:
            print(f"Using custom map: {current_map}")
        
        print(f"Connected! Map: {self.world.get_map().name}")
        
        # Print available spawn points for debugging
        spawn_points = self.world.get_map().get_spawn_points()
        print(f"Available spawn points: {len(spawn_points)}")

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.route_planner = RoutePlanner(
            self.world.get_map(), sampling_resolution=2.0)
        print("Route planner ready (synchronous mode)")

    def spawn_vehicle(self, spawn_index=None):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        print(f"Available spawn points: {len(spawn_points)}")

        if spawn_index is not None and spawn_index < len(spawn_points):
            spawn_point = spawn_points[spawn_index]
        else:
            spawn_point = random.choice(spawn_points)

        for attempt in range(10):
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                print(f"Vehicle spawned (attempt {attempt + 1})")
                for _ in range(10):
                    self.world.tick()
                return True
            spawn_point = random.choice(spawn_points)
        print("ERROR: Could not spawn vehicle!")
        return False

    # =============================================================
    # TRAFFIC LIGHT DETECTION
    # =============================================================
    def check_traffic_light(self):
        """Check traffic light — only obey if close and actually affecting us"""
        try:
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light is None:
                    return "NONE"
                
                state = traffic_light.get_state()
                
                # Check distance to the traffic light
                tl_location = traffic_light.get_location()
                vehicle_location = self.vehicle.get_location()
                distance = vehicle_location.distance(tl_location)
                
                # Only obey if within 15m (ignore distant lights)
                if distance > 15.0:
                    return "NONE"
                
                # Check if light is roughly ahead of us (not on cross street)
                yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
                fwd_x = math.cos(yaw)
                fwd_y = math.sin(yaw)
                dx = tl_location.x - vehicle_location.x
                dy = tl_location.y - vehicle_location.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0.1:
                    dot = (dx * fwd_x + dy * fwd_y) / dist
                    if dot < 0.3:  # Light is behind or to the side
                        return "NONE"
                
                if state == carla.TrafficLightState.Red:
                    return "RED"
                elif state == carla.TrafficLightState.Yellow:
                    return "YELLOW"
                elif state == carla.TrafficLightState.Green:
                    return "GREEN"
        except Exception:
            pass
        return "NONE"

    # =============================================================
    # OFF-ROAD CHECK
    # =============================================================
    def is_on_road(self):
        """Check if vehicle is on a drivable road"""
        try:
            wp = self.world.get_map().get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if wp is None:
                return False
            dist = self.vehicle.get_location().distance(
                wp.transform.location)
            return dist < 3.5
        except Exception:
            return True

    # =============================================================
    # OBSTACLE DETECTION
    # =============================================================
    def get_obstacle_distance(self, frame_count):
        if frame_count - self.actor_cache_frame >= 5:
            self.actor_cache_frame = frame_count
            ego_loc = self.vehicle.get_location()
            nearby = []
            for actor in self.world.get_actors():
                if actor.id == self.vehicle.id:
                    continue
                if not (actor.type_id.startswith('vehicle.') or
                        actor.type_id.startswith('walker.')):
                    continue
                dist = ego_loc.distance(actor.get_location())
                if dist < 25.0:
                    nearby.append(actor)
            self.cached_nearby_actors = nearby

        ego_transform = self.vehicle.get_transform()
        ego_loc = ego_transform.location
        yaw = math.radians(ego_transform.rotation.yaw)
        fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)

        min_dist = float('inf')
        for actor in self.cached_nearby_actors:
            try:
                actor_loc = actor.get_location()
            except:
                continue
            dx = actor_loc.x - ego_loc.x
            dy = actor_loc.y - ego_loc.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > 20.0 or distance < 0.5:
                continue
            dot = (dx * fwd_x + dy * fwd_y) / distance
            if dot < 0.5:
                continue
            cross = abs(fwd_x * dy - fwd_y * dx)
            if cross > 2.5:
                continue
            if distance < min_dist:
                min_dist = distance
        return min_dist

    # =============================================================
    # NPC TRAFFIC
    # =============================================================
    def spawn_npc_vehicles(self, num_vehicles=15):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [bp for bp in bp_lib.filter('vehicle.*')
                       if int(bp.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.world.get_map().get_spawn_points()

        if len(spawn_points) == 0:
            print("  WARNING: No spawn points for NPC vehicles!")
            return

        random.shuffle(spawn_points)
        ego_loc = self.vehicle.get_location()

        # Limit to available spawn points minus some buffer
        max_possible = max(0, len(spawn_points) - 10)
        num_vehicles = min(num_vehicles, max_possible)

        try:
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(3.0)
            traffic_manager.global_percentage_speed_difference(30.0)
        except Exception as e:
            print(f"  WARNING: Traffic manager failed: {e}")
            return

        # Spawn in small batches to avoid crashes
        batch_size = 10
        count = 0
        for batch_start in range(0, num_vehicles, batch_size):
            batch_end = min(batch_start + batch_size, num_vehicles)
            batch_actors = []

            for sp in spawn_points[count:]:
                if len(batch_actors) >= (batch_end - batch_start):
                    break
                if count >= num_vehicles:
                    break
                if sp.location.distance(ego_loc) < 30.0:
                    continue

                bp = random.choice(vehicle_bps)
                if bp.has_attribute('color'):
                    bp.set_attribute('color', random.choice(
                        bp.get_attribute('color').recommended_values))
                if bp.has_attribute('role_name'):
                    bp.set_attribute('role_name', 'autopilot')

                try:
                    npc = self.world.try_spawn_actor(bp, sp)
                    if npc is not None:
                        batch_actors.append(npc)
                        count += 1
                except Exception:
                    continue

            # Set autopilot for this batch
            for npc in batch_actors:
                try:
                    npc.set_autopilot(True, 8000)
                    self.npc_vehicles.append(npc)
                except Exception:
                    try:
                        npc.destroy()
                    except:
                        pass

            # Let simulation settle between batches
            for _ in range(10):
                self.world.tick()

        print(f"  NPC vehicles spawned: {len(self.npc_vehicles)}/{num_vehicles}")
        for _ in range(10):
            self.world.tick()

    def spawn_pedestrians(self, num_pedestrians=10):
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        for _ in range(5):
            self.world.tick()

        walkers_spawned = []
        for i in range(num_pedestrians * 3):
            if len(walkers_spawned) >= num_pedestrians:
                break
            try:
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                bp = random.choice(walker_bps)
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
                if walker is not None:
                    walkers_spawned.append(walker)
            except Exception:
                continue

        self.npc_walkers = walkers_spawned
        print(f"  Pedestrians spawned: {len(walkers_spawned)}/{num_pedestrians}")
        for _ in range(5):
            self.world.tick()

        walker_controller_bp = bp_lib.find('controller.ai.walker')
        controllers = []
        for walker in self.npc_walkers:
            try:
                ctrl = self.world.try_spawn_actor(
                    walker_controller_bp, carla.Transform(), walker)
                if ctrl is not None:
                    controllers.append(ctrl)
            except Exception:
                continue
        self.walker_controllers = controllers
        for _ in range(5):
            self.world.tick()

        for ctrl in self.walker_controllers:
            try:
                target = self.world.get_random_location_from_navigation()
                if target is not None:
                    ctrl.start()
                    ctrl.go_to_location(target)
                    ctrl.set_max_speed(1.0 + random.random() * 1.0)
            except Exception:
                pass
        print(f"  Walker controllers: {len(self.walker_controllers)}")
        for _ in range(5):
            self.world.tick()

    # =============================================================
    # Sensors
    # =============================================================
    def setup_camera(self):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=2.0, y=0.0, z=1.4)),
            attach_to=self.vehicle)
        self.camera.listen(self._camera_callback)
        print("Camera attached (800x600, FOV 100)")

    def setup_collision_sensor(self):
        bp_lib = self.world.get_blueprint_library()
        col_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._collision_callback)
        print("Collision sensor attached")

    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((600, 800, 4))[:, :, :3].copy()
        with self.image_lock:
            self.latest_image = array

    def _collision_callback(self, event):
        now = time.time()
        actor_type = event.other_actor.type_id
        
        # 3-second cooldown per actor type to prevent inflated counts
        last_time = self.collision_cooldown.get(actor_type, 0)
        if now - last_time < 3.0:
            return  # Skip duplicate collision
        
        self.collision_cooldown[actor_type] = now
        self.had_collision = True
        self.collision_time = now
        self.collision_count += 1
        self.metrics.add_collision(actor_type)
        if now - self.last_collision_print > 1.0:
            print(f"  ⚠ COLLISION with {actor_type} "
                  f"(total: {self.collision_count})")
            self.last_collision_print = now

    # =============================================================
    # Model Inference
    # =============================================================
    def preprocess_image(self, image):
        img = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.transform(img)
        return img.unsqueeze(0).to(self.device)

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6

    def predict_controls(self, image, speed_kmh, command_idx):
        img_tensor = self.preprocess_image(image)
        speed_normalized = min(speed_kmh / self.SPEED_NORM_FACTOR, 1.0)
        speed_tensor = torch.tensor(
            [speed_normalized], dtype=torch.float32).to(self.device)
        command_tensor = torch.tensor(
            [command_idx], dtype=torch.long).to(self.device)
        with torch.no_grad():
            pred_controls, pred_speed = self.model(
                img_tensor, speed_tensor, command_tensor)
        return (pred_controls[0, 0].item(), pred_controls[0, 1].item(),
                pred_controls[0, 2].item(),
                pred_speed[0].item() * self.SPEED_NORM_FACTOR)

    # =============================================================
    # Control
    # =============================================================
    def smooth_steering(self, steer):
        self.steer_history.append(steer)
        if len(self.steer_history) < 2:
            return steer
        weights = np.array(
            [0.1, 0.15, 0.2, 0.25, 0.3][-len(self.steer_history):])
        weights = weights / weights.sum()
        return float(np.average(list(self.steer_history), weights=weights))

    def smooth_throttle(self, gas):
        self.throttle_history.append(gas)
        if len(self.throttle_history) < 2:
            return gas
        return float(np.mean(list(self.throttle_history)))

    def apply_control(self, steer, gas, brake, speed_kmh, cmd_idx,
                      steer_hint, obs_dist, traffic_light):
        control = carla.VehicleControl()
        at_intersection = cmd_idx in [1, 2, 3]

        TARGET_SPEED = 35.0
        MAX_SPEED = 45.0
        INTERSECTION_SPEED = 18.0
        CURVE_SPEED = 22.0

        # === DETECT CURVE (relaxed threshold) ===
        steer_magnitude = abs(steer)
        hint_magnitude = abs(steer_hint)
        in_curve = steer_magnitude > 0.25 or hint_magnitude > 0.25

        if in_curve:
            curve_factor = max(steer_magnitude, hint_magnitude)
            current_target = max(15.0, CURVE_SPEED - curve_factor * 15.0)
        elif at_intersection:
            current_target = INTERSECTION_SPEED
        else:
            current_target = TARGET_SPEED

        # === BRAKING DISTANCES ===
        speed_factor = max(1.0, speed_kmh / 15.0)
        HARD_BRAKE_DIST = 8.0 * speed_factor
        SLOW_DIST = 16.0 * speed_factor
        CAUTION_DIST = 25.0 * speed_factor
        # === TRAFFIC LIGHT HANDLING ===
        if traffic_light == "RED":
            control.steer = self.smooth_steering(steer)
            control.throttle = 0.0
            control.brake = 0.8
            control.hand_brake = False
            control.manual_gear_shift = False
            if not self.waiting_for_red:
                self.metrics.red_light_stops += 1
                self.waiting_for_red = True
            self.vehicle.apply_control(control)
            return control, "RED LIGHT"

        if traffic_light == "YELLOW" and speed_kmh < 30.0:
            control.steer = self.smooth_steering(steer)
            control.throttle = 0.0
            control.brake = 0.5
            control.hand_brake = False
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)
            return control, "YELLOW"

        self.waiting_for_red = False
        self.red_light_clear_time = time.time()

        # === OVERTAKE / REVERSE LOGIC ===
        should_overtake, ov_steer, ov_throttle, ov_status = \
            self.attempt_overtake(obs_dist, speed_kmh, traffic_light)

        if should_overtake:
            if ov_status == "REVERSE":
                # Apply reverse control
                hint = self.route_planner.get_next_waypoint_direction(
                    self.vehicle.get_transform())
                control.steer = float(np.clip(-hint * 0.3, -0.5, 0.5))
                control.throttle = 0.4
                control.brake = 0.0
                control.reverse = True
                control.hand_brake = False
                control.manual_gear_shift = False
                self.vehicle.apply_control(control)
                return control, "REVERSE"
            else:
                hint = self.route_planner.get_next_waypoint_direction(
                    self.vehicle.get_transform())
                final_steer = ov_steer + hint * 0.2

                control.steer = float(np.clip(
                    self.smooth_steering(final_steer), -0.5, 0.5))
                control.throttle = ov_throttle
                control.brake = 0.0
                control.hand_brake = False
                control.manual_gear_shift = False
                self.vehicle.apply_control(control)
                return control, ov_status

        # === OBSTACLE SAFETY ===
        if obs_dist < HARD_BRAKE_DIST:
            brake_force = max(0.3, 1.0 - (obs_dist / HARD_BRAKE_DIST))
            control.steer = self.smooth_steering(steer)
            control.throttle = 0.0
            control.brake = brake_force
            control.hand_brake = False
            control.manual_gear_shift = False
            self.waiting_for_traffic = True
            if self.traffic_wait_start is None:
                self.traffic_wait_start = time.time()
            if self.obstacle_wait_start is None:
                self.obstacle_wait_start = time.time()
            self.metrics.obstacle_brakes += 1
            self.vehicle.apply_control(control)
            return control, f"BRAKE({obs_dist:.1f}m)"
        elif obs_dist < SLOW_DIST:
            slow_factor = (obs_dist - HARD_BRAKE_DIST) / max(0.1, SLOW_DIST - HARD_BRAKE_DIST)
            gas = min(gas, 0.15 + slow_factor * 0.2)
            self.waiting_for_traffic = True
            if self.traffic_wait_start is None:
                self.traffic_wait_start = time.time()
            if self.obstacle_wait_start is None:
                self.obstacle_wait_start = time.time()
        elif obs_dist < CAUTION_DIST:
            gas = min(gas, 0.4)
            self.waiting_for_traffic = False
            self.traffic_wait_start = None
            self.obstacle_wait_start = None
        else:
            self.waiting_for_traffic = False
            self.traffic_wait_start = None
            self.obstacle_wait_start = None

        # INTERSECTION HANDLING
        if at_intersection and brake > 0.3 and obs_dist > HARD_BRAKE_DIST:
            brake = 0.0
            gas = max(gas, 0.45)
            if abs(steer_hint) > 0.05:
                steer = 0.4 * steer + 0.6 * steer_hint

        steer = self.smooth_steering(steer)
        if at_intersection and abs(steer_hint) > 0.05:
            steer = 0.6 * steer + 0.4 * steer_hint

        control.steer = float(np.clip(steer, -1.0, 1.0))
        gas = self.smooth_throttle(gas)
        gas = float(np.clip(gas, 0.0, 0.9))
        brake = float(np.clip(brake, 0.0, 1.0))

        # STOPPED TOO LONG
        if speed_kmh < 1.0 and not self.waiting_for_traffic:
            if self.stopped_start_time is None:
                self.stopped_start_time = time.time()
            stopped_duration = time.time() - self.stopped_start_time
            if stopped_duration > 3.0:
                control.throttle = 0.7
                control.brake = 0.0
                if abs(steer_hint) > 0.05:
                    control.steer = float(
                        np.clip(steer_hint * 0.5, -0.5, 0.5))
                if stopped_duration > 6.0:
                    control.throttle = 0.85
                self.vehicle.apply_control(control)
                return control, "UNSTICK"
        elif speed_kmh >= 1.0:
            self.stopped_start_time = None

        # === SPEED CONTROL WITH CURVE AWARENESS ===
        if in_curve and speed_kmh > current_target + 8.0:
            control.throttle = 0.0
            control.brake = 0.4
        elif in_curve and speed_kmh > current_target + 3.0:
            control.throttle = 0.0
            control.brake = 0.2
        elif speed_kmh > MAX_SPEED + 10.0:
            control.throttle = 0.0
            control.brake = 0.9
        elif speed_kmh > MAX_SPEED + 5.0:
            control.throttle = 0.0
            control.brake = 0.6
        elif speed_kmh > MAX_SPEED:
            control.throttle = 0.0
            control.brake = 0.4
        elif speed_kmh > current_target + 5.0:
            control.throttle = 0.0
            control.brake = 0.15
        elif speed_kmh > current_target:
            control.throttle = 0.1
            control.brake = 0.0
        elif speed_kmh < current_target * 0.4:
            control.throttle = max(gas, 0.8)
            control.brake = 0.0
        elif speed_kmh < current_target * 0.7:
            control.throttle = max(gas, 0.6)
            control.brake = 0.0
        elif speed_kmh < current_target:
            speed_deficit = (current_target - speed_kmh) / current_target
            min_throttle = 0.3 + speed_deficit * 0.35
            control.throttle = max(gas, min_throttle)
            control.brake = 0.0
        else:
            control.throttle = gas
            control.brake = 0.0

        control.hand_brake = False
        control.manual_gear_shift = False
        self.vehicle.apply_control(control)
        return control, "OK"
    def can_overtake(self, direction="left"):
        """Check if adjacent lane in given direction is clear"""
        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            yaw = math.radians(vehicle_transform.rotation.yaw)

            current_wp = self.world.get_map().get_waypoint(
                vehicle_location, project_to_road=True,
                lane_type=carla.LaneType.Driving)

            if current_wp is None:
                return False

            if direction == "left":
                adjacent_wp = current_wp.get_left_lane()
            else:
                adjacent_wp = current_wp.get_right_lane()

            if adjacent_wp is None:
                return False

            if adjacent_wp.lane_type != carla.LaneType.Driving:
                return False

            # Check same direction (same sign lane_id = same direction)
            if (current_wp.lane_id * adjacent_wp.lane_id) < 0:
                return False

            fwd_x = math.cos(yaw)
            fwd_y = math.sin(yaw)

            if direction == "left":
                lat_x = -math.sin(yaw)
                lat_y = math.cos(yaw)
            else:
                lat_x = math.sin(yaw)
                lat_y = -math.cos(yaw)

            lane_center_x = vehicle_location.x + lat_x * 3.5
            lane_center_y = vehicle_location.y + lat_y * 3.5

            for actor in self.cached_nearby_actors:
                try:
                    actor_loc = actor.get_location()
                except:
                    continue

                dx = actor_loc.x - lane_center_x
                dy = actor_loc.y - lane_center_y

                lateral_dist = abs(dx * lat_x + dy * lat_y)
                if lateral_dist > 2.5:
                    continue

                longitudinal = dx * fwd_x + dy * fwd_y
                if -5.0 < longitudinal < 30.0:
                    return False

            return True

        except Exception:
            return False

    def get_overtake_steer(self, direction, phase):
        """Get steering for overtake maneuver"""
        if phase == "change":
            return -0.25 if direction == "left" else 0.25
        elif phase == "return":
            return 0.2 if direction == "left" else -0.2
        return 0.0

    def attempt_overtake(self, obs_dist, speed_kmh, traffic_light):
        """
        Overtake state machine with reverse capability.
        Returns: (should_overtake, steer_override, throttle_override, status)
        """
        if traffic_light == "RED":
            self.overtake_state = "NONE"
            self.obstacle_wait_start = None
            return False, 0.0, 0.0, ""

        if self.overtake_state == "NONE":
            if obs_dist < 10.0 and speed_kmh < 3.0:
                # Grace period after red light - give traffic time to move
                if time.time() - self.red_light_clear_time < 10.0:
                    self.obstacle_wait_start = None
                    return False, 0.0, 0.0, "POST_RED_WAIT"
                if self.obstacle_wait_start is None:
                    self.obstacle_wait_start = time.time()

                wait_time = time.time() - self.obstacle_wait_start

                if wait_time > self.obstacle_wait_threshold:
                    if self.can_overtake("left"):
                        self.overtake_state = "LEFT"
                        self.overtake_start_time = time.time()
                        print(f"  🔄 OVERTAKING LEFT (waited {wait_time:.1f}s)")
                        return True, -0.25, 0.5, "OVERTAKE_L"
                    elif self.can_overtake("right"):
                        self.overtake_state = "RIGHT"
                        self.overtake_start_time = time.time()
                        print(f"  🔄 OVERTAKING RIGHT (waited {wait_time:.1f}s)")
                        return True, 0.25, 0.5, "OVERTAKE_R"
                    elif wait_time > 8.0:
                        # Can't change lane — try reversing
                        self.overtake_state = "REVERSE"
                        self.overtake_start_time = time.time()
                        print(f"  🔙 REVERSING (waited {wait_time:.1f}s, no lane)")
                        return True, 0.0, 0.0, "REVERSE"
                    else:
                        return False, 0.0, 0.0, "WAITING"
            else:
                self.obstacle_wait_start = None

            return False, 0.0, 0.0, ""

        elif self.overtake_state == "REVERSE":
            elapsed = time.time() - self.overtake_start_time

            if elapsed < 3.0:
                # Reverse straight back
                hint = self.route_planner.get_next_waypoint_direction(
                    self.vehicle.get_transform())
                return True, -hint * 0.3, 0.0, "REVERSE"

            elif elapsed < 5.0:
                # After reversing, try lane change again
                if self.can_overtake("left"):
                    self.overtake_state = "LEFT"
                    self.overtake_start_time = time.time()
                    print(f"  🔄 After reverse, OVERTAKING LEFT")
                    return True, -0.25, 0.5, "OVERTAKE_L"
                elif self.can_overtake("right"):
                    self.overtake_state = "RIGHT"
                    self.overtake_start_time = time.time()
                    print(f"  🔄 After reverse, OVERTAKING RIGHT")
                    return True, 0.25, 0.5, "OVERTAKE_R"
                else:
                    # Keep reversing
                    return True, 0.0, 0.0, "REVERSE"

            else:
                # Give up reversing, teleport
                print(f"  ⚠ Reverse failed, teleporting...")
                self._teleport_to_nearest_road()
                self.overtake_state = "NONE"
                self.obstacle_wait_start = None
                return False, 0.0, 0.0, ""

        elif self.overtake_state in ["LEFT", "RIGHT"]:
            elapsed = time.time() - self.overtake_start_time
            direction = "left" if self.overtake_state == "LEFT" else "right"

            if elapsed < 2.0:
                steer = self.get_overtake_steer(direction, "change")
                steer *= max(0.3, 1.0 - elapsed / 2.0)
                return True, steer, 0.6, f"OVERTAKE_{direction[0].upper()}"

            elif elapsed < 5.0:
                hint = self.route_planner.get_next_waypoint_direction(
                    self.vehicle.get_transform())
                return True, hint * 0.3, 0.6, "PASSING"

            elif elapsed < 7.0:
                steer = self.get_overtake_steer(direction, "return")
                steer *= max(0.3, 1.0 - (elapsed - 5.0) / 2.0)
                return True, steer, 0.5, "RETURNING"

            else:
                print(f"  ✓ Overtake complete ({elapsed:.1f}s)")
                self.overtake_state = "NONE"
                self.overtake_start_time = None
                self.obstacle_wait_start = None
                return False, 0.0, 0.0, ""

        return False, 0.0, 0.0, ""

    # =============================================================
    # Recovery
    # =============================================================
    def collision_recovery(self):
        self.consecutive_collision_recoveries += 1
        if self.consecutive_collision_recoveries >= 5:
            print(f"  Collision loop "
                  f"({self.consecutive_collision_recoveries}x). Teleporting...")
            self._teleport_to_nearest_road()
            self.consecutive_collision_recoveries = 0
            self.had_collision = False
            self.steer_history.clear()
            self.throttle_history.clear()
            return

        print(f"  Recovery #{self.consecutive_collision_recoveries}...")
        self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(6):
            self.world.tick()

        steer_dir = random.choice([-0.5, 0.5, -0.3, 0.3, 0.0])
        rc = carla.VehicleControl()
        rc.throttle = 0.5
        rc.steer = steer_dir
        rc.reverse = True
        self.vehicle.apply_control(rc)
        for _ in range(40):
            self.world.tick()

        self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(6):
            self.world.tick()
        self.had_collision = False
        self.steer_history.clear()
        self.throttle_history.clear()
        self.stopped_start_time = None

    def _teleport_to_nearest_road(self):
        try:
            current_loc = self.vehicle.get_location()
            wp = self.world.get_map().get_waypoint(
                current_loc, project_to_road=True)
            if not wp:
                return

            # Try multiple waypoints ahead to find a clear spot
            candidates = []
            test_wp = wp
            for i in range(10):
                next_wps = test_wp.next(10.0)
                if not next_wps:
                    break
                test_wp = next_wps[0]
                candidates.append(test_wp)

            # Also try waypoints behind
            test_wp = wp
            for i in range(5):
                prev_wps = test_wp.previous(10.0)
                if not prev_wps:
                    break
                test_wp = prev_wps[0]
                candidates.append(test_wp)

            # Find candidate with no nearby vehicles
            best_wp = None
            best_min_dist = 0

            all_vehicles = [a for a in self.world.get_actors()
                            if a.type_id.startswith('vehicle.')
                            and a.id != self.vehicle.id]

            for cand_wp in candidates:
                cand_loc = cand_wp.transform.location
                min_dist = float('inf')
                for v in all_vehicles:
                    d = cand_loc.distance(v.get_location())
                    if d < min_dist:
                        min_dist = d
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_wp = cand_wp

            if best_wp is None or best_min_dist < 8.0:
                # Fallback: just go far ahead
                test_wp = wp
                for i in range(20):
                    next_wps = test_wp.next(15.0)
                    if not next_wps:
                        break
                    test_wp = next_wps[0]
                best_wp = test_wp

            new_loc = best_wp.transform.location
            new_loc.z += 0.5
            self.vehicle.set_transform(
                carla.Transform(new_loc, best_wp.transform.rotation))
            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(20):
                self.world.tick()
            self.vehicle.apply_control(carla.VehicleControl())
            for _ in range(5):
                self.world.tick()

            self.steer_history.clear()
            self.throttle_history.clear()
            self.stopped_start_time = None
            self.had_collision = False
            self.consecutive_collision_recoveries = 0
            self.overtake_state = "NONE"
            self.obstacle_wait_start = None
            self.waiting_for_traffic = False
            self.traffic_wait_start = None
            print(f"  Teleported to ({new_loc.x:.0f}, {new_loc.y:.0f}) "
                  f"[clearance: {best_min_dist:.0f}m]")
        except Exception as e:
            print(f"  Teleport failed: {e}")

    def is_stuck(self, vehicle_location):
        now = time.time()
        self.position_history.append(
            (now, vehicle_location.x, vehicle_location.y))
        if len(self.position_history) < 10:
            return False
        if now - self.position_history[0][0] < 15.0:
            return False

        if self.waiting_for_traffic:
            if (self.traffic_wait_start and
                    (now - self.traffic_wait_start) > 25.0):
                print(f"  [Stuck] Waited 25s+ for traffic")
                self.position_history.clear()
                self.waiting_for_traffic = False
                self.traffic_wait_start = None
                return True
            return False

        target_time = now - 15.0
        best_entry = self.position_history[0]
        for entry in self.position_history:
            if entry[0] <= target_time:
                best_entry = entry
            else:
                break
        dx = vehicle_location.x - best_entry[1]
        dy = vehicle_location.y - best_entry[2]
        dist_moved = math.sqrt(dx * dx + dy * dy)
        if dist_moved < 3.0:
            print(f"  [Stuck] Moved {dist_moved:.1f}m in "
                  f"{now - best_entry[0]:.1f}s")
            self.position_history.clear()
            return True
        return False

    # =============================================================
    # Route
    # =============================================================
    def plan_route(self, destination_index=None):
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_location = self.vehicle.get_location()
        if destination_index is not None:
            destination = spawn_points[destination_index].location
        else:
            best_dest, best_dist = None, 0
            for _ in range(30):
                dest_sp = random.choice(spawn_points)
                dist = vehicle_location.distance(dest_sp.location)
                if 80 < dist < 300 and dist > best_dist:
                    best_dest = dest_sp.location
                    best_dist = dist
            destination = best_dest or random.choice(spawn_points).location
        print(f"  Destination: ({destination.x:.0f}, {destination.y:.0f})")
        self.metrics.routes_attempted += 1
        return self.route_planner.set_route(vehicle_location, destination)

    # =============================================================
    # Main Loop
    # =============================================================
    def run(self, duration=120, spawn_index=0, destination_index=None,
        num_vehicles=15, num_pedestrians=10, show_hud=True,
        use_custom_map=False):
        print("\n" + "=" * 60)
        print("CILRS AUTONOMOUS DRIVING V4.0")
        print("Traffic Lights | Dashboard HUD | Evaluation Metrics")
        print("=" * 60)

        try:
            self.connect(use_custom_map=use_custom_map)
            if not self.spawn_vehicle(spawn_index):
                return

            self.setup_camera()
            self.setup_collision_sensor()

            print("Waiting for camera...")
            for _ in range(50):
                self.world.tick()
                if self.latest_image is not None:
                    break
            if self.latest_image is None:
                print("ERROR: No camera image!")
                return

            print("\nSpawning traffic...")
            if num_vehicles > 0:
                self.spawn_npc_vehicles(num_vehicles)
            if num_pedestrians > 0:
                self.spawn_pedestrians(num_pedestrians)

            self.plan_route(destination_index)

            print(f"\nDriving for {duration}s | "
                  f"{len(self.npc_vehicles)} vehicles | "
                  f"{len(self.npc_walkers)} pedestrians")
            if show_hud:
                print("Dashboard HUD: ON (press ESC to close HUD)")
            print("-" * 70)

            self.running = True
            self.metrics.start()
            start_time = time.time()
            frame_count = 0
            route_count = 0
            last_print_time = 0
            last_loop_time = time.time()

            while self.running and (time.time() - start_time) < duration:

                if self.had_collision and \
                        (time.time() - self.collision_time) < 3.0:
                    self.collision_recovery()
                    last_loop_time = time.time()
                    continue

                with self.image_lock:
                    if self.latest_image is None:
                        self.world.tick()
                        continue
                    image = self.latest_image.copy()
                image = image[:, :, ::-1].copy()

                speed_kmh = self.get_vehicle_speed()
                vehicle_transform = self.vehicle.get_transform()
                vehicle_location = vehicle_transform.location

                # Calculate dt for metrics
                now = time.time()
                dt = now - last_loop_time
                last_loop_time = now

                # Checks
                obs_dist = self.get_obstacle_distance(frame_count)
                traffic_light = self.check_traffic_light()
                on_road = self.is_on_road()
                if not on_road:
                    self.off_road_frames_consecutive += 1
                    if self.off_road_frames_consecutive > 10:
                        print(f"  ⚠ OFF-ROAD for {self.off_road_frames_consecutive} frames! Recovering...")
                        self._teleport_to_nearest_road()
                        self.off_road_frames_consecutive = 0
                        continue
                else:
                    self.off_road_frames_consecutive = 0

                # Update metrics
                self.metrics.update(speed_kmh,
                                    self.steer_history[-1]
                                    if self.steer_history else 0.0,
                                    on_road, dt)

                if self.is_stuck(vehicle_location):
                    print("  ⚠ STUCK! Teleporting...")
                    self._teleport_to_nearest_road()
                    self.plan_route()
                    self.stopped_start_time = None
                    continue

                cmd_idx, cmd_name = self.route_planner.get_command(
                    vehicle_location)
                steer_hint = \
                    self.route_planner.get_next_waypoint_direction(
                        vehicle_transform)

                if self.route_planner.is_route_complete(vehicle_location):
                    route_count += 1
                    self.metrics.routes_completed += 1
                    print(f"\n  ✓ Route {route_count} complete!")
                    self.consecutive_collision_recoveries = 0
                    self.plan_route()

                steer, gas, brake, pred_speed = self.predict_controls(
                    image, speed_kmh, cmd_idx)

                control, status = self.apply_control(
                    steer, gas, brake, speed_kmh, cmd_idx,
                    steer_hint, obs_dist, traffic_light)

                frame_count += 1
                elapsed = time.time() - start_time

                # Spectator camera
                spectator = self.world.get_spectator()
                car_t = self.vehicle.get_transform()
                yaw = math.radians(car_t.rotation.yaw)
                spectator.set_transform(carla.Transform(
                    carla.Location(
                        x=car_t.location.x - 8 * math.cos(yaw),
                        y=car_t.location.y - 8 * math.sin(yaw),
                        z=car_t.location.z + 5),
                    carla.Rotation(pitch=-20,
                                  yaw=car_t.rotation.yaw)))

                # Dashboard HUD
                if show_hud:
                    remaining = self.route_planner.distance_remaining(
                        vehicle_location)
                    hud_continue = self.hud.update(
                        image, speed_kmh, cmd_name,
                        control.steer, control.throttle, control.brake,
                        obs_dist, traffic_light, remaining,
                        status, self.metrics)
                    if not hud_continue:
                        show_hud = False
                        self.hud.close()
                        print("  [HUD closed by user]")

                # Terminal status
                if elapsed - last_print_time >= 2.0:
                    remaining = self.route_planner.distance_remaining(
                        vehicle_location)
                    obs_str = f"{obs_dist:.0f}m" if obs_dist < 20 \
                        else "clear"
                    tl_str = f" 🔴" if traffic_light == "RED" \
                        else (f" 🟡" if traffic_light == "YELLOW" else "")
                    print(f"  [{elapsed:5.0f}s] {cmd_name:8s} | "
                          f"Spd:{speed_kmh:4.0f} | "
                          f"Str:{control.steer:+.2f} | "
                          f"Dst:{remaining:4.0f}m | "
                          f"Obs:{obs_str} | "
                          f"{status}{tl_str}")
                    last_print_time = elapsed

                self.world.tick()

            # ── End of drive ──
            total_time = time.time() - start_time

            print("\n" + "=" * 60)
            print("DRIVING COMPLETE")
            print("=" * 60)
            print(f"Duration:    {total_time:.1f}s")
            print(f"Frames:      {frame_count}")
            print(f"Avg FPS:     {frame_count / total_time:.1f}")
            print(f"NPC Cars:    {len(self.npc_vehicles)}")
            print(f"Pedestrians: {len(self.npc_walkers)}")

            # Print full evaluation report
            self.metrics.print_report()

        except KeyboardInterrupt:
            print("\nStopped by user")
            self.metrics.print_report()
        finally:
            if show_hud:
                self.hud.close()
            self.cleanup()

    # =============================================================
    # Cleanup
    # =============================================================
    def cleanup(self):
        print("\nCleaning up...")
        self.running = False

        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except:
                pass

        destroy_ids = []
        for ctrl in self.walker_controllers:
            try:
                ctrl.stop()
            except:
                pass
            destroy_ids.append(ctrl.id)
        for walker in self.npc_walkers:
            destroy_ids.append(walker.id)
        for npc in self.npc_vehicles:
            try:
                npc.set_autopilot(False)
            except:
                pass
            destroy_ids.append(npc.id)
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
            except:
                pass
            destroy_ids.append(self.collision_sensor.id)
        if self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
            destroy_ids.append(self.camera.id)
        if self.vehicle is not None:
            destroy_ids.append(self.vehicle.id)

        if destroy_ids and self.client is not None:
            try:
                batch = [carla.command.DestroyActor(x)
                         for x in destroy_ids]
                self.client.apply_batch_sync(batch)
                print(f"Destroyed {len(destroy_ids)} actors")
            except:
                pass

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
        print("Cleanup complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CILRS Autonomous Driving V4.0")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/rohith/carla_simulator/"
                                "model/checkpoint_best.pth")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--destination", type=int, default=None)
    parser.add_argument("--vehicles", type=int, default=15)
    parser.add_argument("--pedestrians", type=int, default=10)
    parser.add_argument("--no-hud", action="store_true",
                        help="Disable dashboard HUD window")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--map", type=str, default=None,
                        help="Use 'cusat' or 'custom' to skip Town01 loading")
    args = parser.parse_args()

    driver = AutonomousDriver(
        checkpoint_path=args.checkpoint, device=args.device)
    driver.run(duration=args.duration,
               spawn_index=args.spawn,
               destination_index=args.destination,
               num_vehicles=args.vehicles,
               num_pedestrians=args.pedestrians,
               show_hud=not args.no_hud,
               use_custom_map=(args.map is not None))
