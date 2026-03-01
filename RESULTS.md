# 📊 Benchmark Results — CILRS Autonomous Driving

Full evaluation results for the CILRS agent on **CARLA Town01** across all 5 weather conditions.  
All runs: spawn point 0 → destination 50, 40 NPC vehicles, 5 pedestrians.

---

## ✅ Weather Benchmark Summary

| Weather | Grade | Score | Collisions | Destination | Time | Max Speed |
|:-------:|:-----:|:-----:|:----------:|:-----------:|:----:|:---------:|
| ☀️ Clear | **A+** | **99.4** | 0 | ✅ | 289s (4.8 min) | 35.5 km/h |
| 🌧️ Rain | **A+** | **98.7** | 0 | ✅ | 384s (6.4 min) | 28.7 km/h |
| 🌫️ Fog | **A+** | **99.3** | 0 | ✅ | 366s (6.1 min) | 30.5 km/h |
| 🌙 Night | **A+** | **99.3** | 0 | ✅ | 373s (6.2 min) | 25.7 km/h |
| ⛈️ Hard Rain | **A+** | **97.8** | 0 | ✅ | 339s (5.6 min) | 21.1 km/h |

**Average Score: 98.9 / 100 | Zero Collisions Across All 5 Conditions**

---

## 📐 Scoring Formula

Scores are computed live by `DrivingMetrics` in `autonomous_drive.py`:

```
Overall  = Safety × 0.6  +  Comfort × 0.3  +  Route Completion × 0.1

Safety   = 100 − (collisions × 15)
               − (red_light_violations × 10)
               − (off_road_percentage × 40)

Comfort  = 100 − (average_steering_jerk × 1000)
         where steering_jerk = |steer[t] − steer[t−1]|  per frame

Route Completion = (routes_completed / routes_attempted) × 100
```

| Grade | Overall Score |
|-------|:-------------:|
| A+ (Excellent) | ≥ 90 |
| A  (Very Good) | ≥ 80 |
| B+ (Good) | ≥ 70 |
| B  (Satisfactory) | ≥ 60 |
| C  (Needs Improvement) | < 60 |

---

## 🔍 Per-Weather Analysis

### ☀️ Clear — Score 99.4

The baseline condition. The agent drives at full target speed (35 km/h), no traction control, standard braking. The 0.6% score deficit from a perfect 100 is accounted for by minor off-road edge cases at tight corners and natural steering jerk during overtake manoeuvres.

| Metric | Value |
|--------|-------|
| Score | 99.4 |
| Completion time | 289s |
| Max speed achieved | 35.5 km/h |
| Average speed | ~23 km/h (includes stops) |
| Collisions | 0 |
| Red light violations | 0 |

---

### 🌧️ Rain — Score 98.7

Speed capped at 28 km/h. Brake factor 1.5× and traction control above 15 km/h engaged. The longer completion time (384s vs 289s) directly reflects the lower permitted speed. Score is marginally lower due to slightly increased steering jerk from the 1.05× steer damping interacting with the model's raw predictions.

| Metric | Value |
|--------|-------|
| Score | 98.7 |
| Completion time | 384s (+33% vs clear) |
| Max speed achieved | 28.7 km/h |
| Collisions | 0 |
| Red light violations | 0 |

---

### 🌫️ Fog — Score 99.3

Speed capped at 30 km/h with reduced lookahead (8m vs 10m in clear). The shorter lookahead means the agent reacts to curves later, requiring slightly sharper corrections, which marginally affects comfort score. No traction control needed — dry road, visibility-limited condition only.

| Metric | Value |
|--------|-------|
| Score | 99.3 |
| Completion time | 366s |
| Max speed achieved | 30.5 km/h |
| Collisions | 0 |
| Red light violations | 0 |

---

### 🌙 Night — Score 99.3

Same parameters as fog (30 km/h cap, 1.3× braking, 8m lookahead). The identical score to fog confirms the parameter parity is appropriate — headlights in CARLA provide sufficient local visibility for the camera-based model, and the main challenge is reduced reaction distance rather than sensor degradation.

| Metric | Value |
|--------|-------|
| Score | 99.3 |
| Completion time | 373s |
| Max speed achieved | 25.7 km/h |
| Collisions | 0 |
| Red light violations | 0 |

---

### ⛈️ Hard Rain — Score 97.8

The most challenging condition. Speed capped at 20 km/h, brake factor 2.0×, steer damping 1.15×, traction control active above 15 km/h. The lower score (97.8 vs 99+ elsewhere) is due to increased steering corrections from the aggressive steer damping fighting the model's raw outputs. Despite this, the agent successfully reached the destination with zero collisions — the safety subscore remained perfect.

| Metric | Value |
|--------|-------|
| Score | 97.8 |
| Completion time | 339s (faster than rain — lower speed cap but fewer hesitations) |
| Max speed achieved | 21.1 km/h |
| Collisions | 0 |
| Red light violations | 0 |

---

## 🤖 Model Training Results

Trained on Kaggle (NVIDIA T4), 20 epochs, 176,256 frames from 5 CARLA Town01 sessions.

| Metric | Value |
|--------|-------|
| Final Val Loss | **0.0538** |
| Steering Correlation | **0.9861** |
| Throttle Correlation | 0.9589 |
| Brake Correlation | 0.9815 |
| Speed Correlation | 0.9844 |

All four control outputs achieved correlation > 0.95, with steering at 0.9861 indicating the model closely replicates human steering decisions across all four command branches (Follow, Left, Right, Straight).

Training curves (`output/result.png`) show:
- Validation loss below training loss from epoch 3 onwards — no overfitting
- Smooth monotonic convergence through epoch 20
- Val Steer Error by Command: all 4 branches converge below 0.005 by epoch 15

---

## 🗺️ CUSAT Custom Map — Experimental Results

> ⚠️ Results on the CUSAT map were **not satisfactory**. The agent showed degraded route-following and erratic steering due to distribution shift — the model was trained exclusively on Town01 data and does not generalise to the OSM-derived road geometry of the CUSAT campus.

This is documented as future work. See [README — CUSAT section](README.md#-custom-map--cusat-campus) for full technical explanation.

---

## 📋 Test Environment

| Parameter | Value |
|-----------|-------|
| Map | CARLA Town01 |
| Spawn point | 0 |
| Destination | 50 |
| NPC vehicles | 40 |
| Pedestrians | 5 |
| Simulator | CARLA 0.9.10 |
| Mode | Synchronous (world.tick()) |
| GPU | NVIDIA RTX 4060 Laptop 8GB |
| OS | Parrot Linux |
| Python | 3.7 (conda: carla_project) |
