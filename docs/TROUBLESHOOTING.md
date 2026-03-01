# 🛠️ Troubleshooting Guide

Common issues encountered during this project and their fixes.
All solutions are confirmed working on **CARLA 0.9.10 / Python 3.7 / Parrot Linux**.

---

## Table of Contents

1. [CARLA Freezes After Script Exits](#1-carla-freezes-after-script-exits)
2. [std::out_of_range Crash When Spawning NPCs](#2-stdout_of_range-crash-when-spawning-npcs)
3. [NumPy _core Module Error on Checkpoint Load](#3-numpy-_core-module-error-on-checkpoint-load)
4. [Camera Callback Thread Crash](#4-camera-callback-thread-crash)
5. [Traffic Manager Port Already in Use](#5-traffic-manager-port-already-in-use)
6. [Agent Brakes for Oncoming Traffic (False Positive)](#6-agent-brakes-for-oncoming-traffic-false-positive)
7. [Agent Ignores Red Lights / Brakes for Wrong Lights](#7-agent-ignores-red-lights--brakes-for-wrong-lights)
8. [Vehicle Stuck with No Obstacles](#8-vehicle-stuck-with-no-obstacles)
9. [CUSAT Map Fails to Load](#9-cusat-map-fails-to-load)
10. [Low FPS / Simulation Running Slowly](#10-low-fps--simulation-running-slowly)
11. [Collision Count Inflated (100+ Collisions Instantly)](#11-collision-count-inflated-100-collisions-instantly)
12. [set_velocity AttributeError](#12-set_velocity-attributeerror)

---

## 1. CARLA Freezes After Script Exits

**Symptom:** After the Python script finishes or crashes, CARLA becomes unresponsive. Mouse clicks do nothing, the CARLA window stops rendering.

**Cause:** The script enabled synchronous mode (`world.tick()`) but did not disable it on exit. CARLA is waiting for a `tick()` call that will never come.

**Fix:** Always reset settings in the `cleanup()` method:

```python
def cleanup(self):
    if self.world is not None:
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except:
            pass
```

Wrap `cleanup()` in a `finally` block so it always runs even on crash:

```python
try:
    # main driving loop
except KeyboardInterrupt:
    pass
finally:
    self.cleanup()
```

---

## 2. std::out_of_range Crash When Spawning NPCs

**Symptom:** CARLA crashes with a C++ `std::out_of_range` exception when spawning many NPC vehicles at once.

**Cause:** Spawning all vehicles in a single batch with `apply_batch_sync()` can exceed internal CARLA limits, especially when spawn points are close to each other or near the ego vehicle.

**Fix:** Spawn in small batches of 10 with simulation ticks between batches, and filter out spawn points within 30m of the ego vehicle:

```python
batch_size = 10
for batch_start in range(0, num_vehicles, batch_size):
    # spawn batch_size vehicles
    # ...
    for _ in range(10):
        self.world.tick()   # let simulation settle between batches
```

Also cap the total vehicles to `len(spawn_points) - 10` to keep a buffer:

```python
max_possible = max(0, len(spawn_points) - 10)
num_vehicles = min(num_vehicles, max_possible)
```

---

## 3. NumPy _core Module Error on Checkpoint Load

**Symptom:**
```
ModuleNotFoundError: No module named 'numpy._core'
```
Occurs when loading a `.pth` checkpoint saved with a newer NumPy version on a Python 3.7 environment.

**Cause:** NumPy 2.x renamed internal modules. PyTorch's checkpoint loader tries to unpickle references to `numpy._core` which does not exist in NumPy 1.x.

**Fix:** Redirect the missing modules before loading:

```python
import sys
import numpy as np

class _NumpyCoreModule:
    def __init__(self):
        import numpy.core.multiarray as ma
        import numpy.core.numeric as nu
        self.multiarray = ma
        self.numeric = nu

sys.modules['numpy._core'] = _NumpyCoreModule()
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric

# Now safe to load
checkpoint = torch.load('checkpoint_best.pth', map_location=device)
```

Place this block at the top of the script, before any `torch.load()` call.

---

## 4. Camera Callback Thread Crash

**Symptom:** Random crashes with errors like `Segmentation fault` or `RuntimeError: dictionary changed size during iteration` during the camera image callback.

**Cause:** CARLA's camera sensor fires its callback on a separate thread. Writing to a shared `image` variable without a lock causes race conditions when the main loop reads it simultaneously.

**Fix:** Use a `threading.Lock` and copy the array:

```python
import threading

self.image_lock = threading.Lock()
self.latest_image = None

def camera_callback(self, image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]   # drop alpha
    with self.image_lock:
        self.latest_image = array.copy()   # .copy() is critical

# In main loop:
with self.image_lock:
    if self.latest_image is None:
        continue
    image = self.latest_image.copy()
```

---

## 5. Traffic Manager Port Already in Use

**Symptom:**
```
RuntimeError: trying to create rpc server on port 8000 but the port is already in use
```

**Cause:** A previous run left the Traffic Manager process alive, or another CARLA instance is running.

**Fix:**

Option A — Kill the leftover process:
```bash
pkill -f CarlaUE4
pkill -f python
# Wait 5s, then restart CARLA
```

Option B — Use a different port:
```python
traffic_manager = self.client.get_trafficmanager(8001)
traffic_manager.set_synchronous_mode(True)
# When setting autopilot, use the same port:
npc.set_autopilot(True, 8001)
```

---

## 6. Agent Brakes for Oncoming Traffic (False Positive)

**Symptom:** The agent slows down or stops on a straight road when there is no obstacle ahead — a vehicle in the opposite lane is triggering the obstacle detector.

**Cause:** The obstacle detection forward-projection check was too wide, catching vehicles in opposing lanes at curves where their lateral offset temporarily drops below the threshold.

**Fix:** Add a heading dot-product check to skip vehicles travelling in the opposite direction:

```python
# Get actor velocity to check direction
actor_vel = actor.get_velocity()
actor_speed = math.sqrt(actor_vel.x**2 + actor_vel.y**2)
if actor_speed > 1.0:
    # dot product of ego forward and actor velocity
    heading_dot = (actor_vel.x * fwd_x + actor_vel.y * fwd_y) / actor_speed
    if heading_dot < -0.5:   # moving toward us — oncoming
        continue
```

Also tighten the lateral threshold from 2.5m to 1.8m for single-lane roads.

---

## 7. Agent Ignores Red Lights / Brakes for Wrong Lights

**Symptom A:** Agent runs through red lights at intersections.  
**Symptom B:** Agent brakes to a stop when a traffic light on a cross street turns red.

**Cause:** `is_at_traffic_light()` returns True for lights on perpendicular roads. Without a direction check, the agent either obeys all nearby lights (symptom B) or the detection fires too far away to matter (symptom A).

**Fix:** Filter by distance AND heading:

```python
def check_traffic_light(self):
    if self.vehicle.is_at_traffic_light():
        tl = self.vehicle.get_traffic_light()
        if tl is None:
            return "NONE"

        # Distance filter
        dist = self.vehicle.get_location().distance(tl.get_location())
        if dist > 15.0:
            return "NONE"

        # Heading filter — only obey lights ahead of us
        yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
        dx = tl.get_location().x - self.vehicle.get_location().x
        dy = tl.get_location().y - self.vehicle.get_location().y
        d = math.sqrt(dx*dx + dy*dy)
        if d > 0.1:
            dot = (dx * fwd_x + dy * fwd_y) / d
            if dot < 0.3:   # light is behind or to the side
                return "NONE"

        # Now check state
        state = tl.get_state()
        if state == carla.TrafficLightState.Red:
            return "RED"
        ...
    return "NONE"
```

---

## 8. Vehicle Stuck with No Obstacles

**Symptom:** The agent stops and never moves again even though the road ahead is clear. Speed shows 0 km/h indefinitely.

**Cause:** The model predicted a high brake value at an unusual road segment, the vehicle stopped, and the low-speed throttle boost wasn't enough to restart it.

**Fix:** Implement a stopped-time escalation:

```python
if speed_kmh < 1.0 and not self.waiting_for_traffic:
    if self.stopped_start_time is None:
        self.stopped_start_time = time.time()
    stopped = time.time() - self.stopped_start_time

    if stopped > 3.0:
        control.throttle = 0.7
        control.brake = 0.0
    if stopped > 6.0:
        control.throttle = 0.85   # harder push
    if stopped > 10.0:
        self._teleport_to_nearest_road()   # last resort
```

The `is_stuck()` method also checks location change over 30 frames as a secondary trigger for teleport.

---

## 9. CUSAT Map Fails to Load

**Symptom:** `load_cusat.py` raises an error or CARLA crashes when loading the custom `.xodr` file.

**Common errors and fixes:**

**Error:** `generate_opendrive_world() returned False`  
**Fix:** The `.xodr` file has geometry errors. Regenerate with tighter parameters:
```python
xodr_content = open('cusat_campus.xodr').read()
vertex_distance = 2.0      # default 1.0 is too dense for OSM roads
max_road_length = 500.0
wall_height = 0.0
additional_width = 0.6
smooth_junctions = True
enable_mesh_visibility = True
world = client.generate_opendrive_world(
    xodr_content,
    carla.OpendriveGenerationParameters(
        vertex_distance, max_road_length,
        wall_height, additional_width,
        smooth_junctions, enable_mesh_visibility
    )
)
```

**Error:** `No spawn points available`  
**Fix:** OSM data may have disconnected road segments. Check the `.osm` source for gaps around roundabouts or campus gates and manually connect them before converting.

**Error:** Simulation loads but vehicle falls through the ground  
**Fix:** Increase `z` offset when spawning: `spawn_transform.location.z += 1.0`

---

## 10. Low FPS / Simulation Running Slowly

**Symptom:** The driving loop runs at 2–3 FPS instead of the expected 10–15 FPS.

**Fixes (in order of impact):**

1. Start CARLA with `-quality-level=Low`:
   ```bash
   ./CarlaUE4.sh -opengl -quality-level=Low
   ```

2. Close the CARLA spectator window rendering or move it off-screen.

3. Reduce NPC count. Each NPC vehicle adds CPU load. Drop from 40 to 20 for testing.

4. Use `fixed_delta_seconds = 0.1` (10 FPS cap) in synchronous mode to prevent the simulation from trying to run faster than the inference loop:
   ```python
   settings.fixed_delta_seconds = 0.1
   settings.synchronous_mode = True
   world.apply_settings(settings)
   ```

5. Ensure the model is running on GPU:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

---

## 11. Collision Count Inflated (100+ Collisions Instantly)

**Symptom:** The collision sensor fires 50–100 times per second on spawn, reporting hundreds of collisions before the car has moved.

**Cause:** CARLA's collision sensor fires repeatedly for a single physical contact event. On spawn, if the vehicle overlaps slightly with the ground mesh, it can fire continuously.

**Fix:** Implement a per-actor cooldown:

```python
self.collision_cooldowns = {}   # actor_id -> last_collision_time

def on_collision(self, event):
    actor_id = event.other_actor.id
    now = time.time()
    if now - self.collision_cooldowns.get(actor_id, 0) < 3.0:
        return   # ignore repeat collisions within 3s for same actor
    self.collision_cooldowns[actor_id] = now
    actor_type = event.other_actor.type_id.split('.')[0]
    self.metrics.add_collision(actor_type)
```

---

## 12. set_velocity AttributeError

**Symptom:**
```
AttributeError: 'Actor' object has no attribute 'set_velocity'
```

**Cause:** The API method was renamed between CARLA versions.

**Fix:** Use `set_target_velocity()` in CARLA 0.9.10:

```python
# Wrong (older API):
vehicle.set_velocity(carla.Vector3D(x=5.0, y=0, z=0))

# Correct for CARLA 0.9.10:
vehicle.set_target_velocity(carla.Vector3D(x=5.0, y=0, z=0))
```

---

## General Tips

- Always start CARLA before running any Python script — the client will hang on `carla.Client()` if the server isn't up.
- After any crash, send `Ctrl+C` and wait for `cleanup()` to complete before restarting. If CARLA is frozen, kill with `pkill -f CarlaUE4`.
- Run `nvidia-smi` to confirm the GPU is being used by both CARLA (`CarlaUE4`) and Python (`python3`).
- If the simulation desyncs (vehicles teleporting wildly), increase the `fixed_delta_seconds` from 0.05 to 0.1 to give the physics solver more time per tick.
