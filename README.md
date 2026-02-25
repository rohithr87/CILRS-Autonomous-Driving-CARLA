# CILRS Autonomous Driving in CARLA Simulator

> **B.Tech Computer Science & Engineering — Major Project**
> End-to-end autonomous driving using Conditional Imitation Learning with a ResNet-34 backbone (CILRS) in the CARLA 0.9.10 simulator.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Results & Performance](#results--performance)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Dataset & Training](#dataset--training)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Custom Map — CUSAT Campus](#custom-map--cusat-campus)
- [Tech Stack](#tech-stack)
- [References](#references)
- [Author](#author)

---

## Project Overview

This project implements an **end-to-end autonomous driving agent** using Conditional Imitation Learning with ResNet (CILRS). The agent learns to drive in a simulated urban environment solely from human demonstrations — no hand-crafted rules, no reward engineering.

Given a camera image, current speed, and a high-level navigation command (Follow Lane / Turn Left / Turn Right / Go Straight), the neural network directly predicts steering, throttle, and brake to control the vehicle in real time.

The project was evaluated in CARLA Town01 with 40 NPC vehicles and 5 pedestrians, achieving a **Safety Score of 99.8/100** with zero collisions over a 450-second run.

---

## Demo

### Simulation in Action — CARLA Town01

**Normal driving at 25 km/h — FOLLOW command, clear road, 0 collisions:**

![CILRS driving — normal conditions](demo/output1.png)

**Stopped at red light — braking correctly, CAUTION alert on HUD:**

![CILRS driving — red light stop](demo/output2.png)

> Both screenshots show the real-time dashboard HUD (top-right), the third-person simulator view (left), and the terminal log of per-second driving metrics (bottom).

### Training Results — Loss Curves (Kaggle, NVIDIA T4)

![CILRS Training Results — Best Epoch 20, Val Loss 0.0538](demo/result.png)

All six plots show healthy convergence — validation loss consistently below training loss from epoch 3 onwards, with no signs of overfitting. The **Val Steer Error by Command** panel (bottom-right) confirms balanced performance across all four navigation branches (Follow, Left, Right, Straight), with errors converging below 0.005 by epoch 15.

---

## Results & Performance

### Simulation Evaluation (Town01)

| Metric                | Value         |
|-----------------------|---------------|
| Safety Score          | **99.8 / 100**|
| Overall Grade         | **A+ (Excellent)** |
| Total Collisions      | 0             |
| Red Light Violations  | 0             |
| Steering Correlation  | 0.9861        |
| Max Speed Achieved    | 35.4 km/h     |
| Off-road Percentage   | 0.4%          |
| Test Duration         | 450 seconds   |
| NPC Vehicles          | 40            |
| Pedestrians           | 5             |

### Training Metrics

| Metric                | Value    |
|-----------------------|----------|
| Validation Loss       | 0.0538   |
| Steering Correlation  | 0.9861   |
| Throttle Correlation  | 0.9589   |
| Brake Correlation     | 0.9815   |
| Speed Correlation     | 0.9844   |

> Training curves showed rapid convergence within the first 5 epochs, with smooth loss reduction across all control outputs through epoch 20. The "Val Steer Error by Command" plot confirmed balanced performance across all four navigation branches.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      CILRS Model                         │
│                                                          │
│  Camera Image (200×88) ──► ResNet-34 ──► Visual Features │
│                                               │          │
│  Speed (normalized) ──► Speed Encoder ──►  Concat        │
│                                               │          │
│  Navigation Command ──────────────────► Branch Selector  │
│                                               │          │
│                              ┌────────────────┴────────┐ │
│                              │   4 Conditional Branches│ │
│                              │  Follow / Left /        │ │
│                              │  Right / Straight       │ │
│                              └────────────────┬────────┘ │
│                                               │          │
│                              Predicted Controls          │
│                         [Steering, Throttle, Brake]      │
│                                               │          │
│                         Speed Prediction (Auxiliary)     │
└──────────────────────────────────────────────────────────┘
```

### Model Details

| Component         | Details                                    |
|-------------------|--------------------------------------------|
| Backbone          | ResNet-34 (pretrained on ImageNet)         |
| Total Parameters  | 22.4 Million                               |
| Input Image Size  | 200 × 88 (RGB)                             |
| Navigation Branches | 4 — Follow, Left, Right, Straight        |
| Loss Functions    | MSE (controls) + MSE (speed auxiliary)     |
| Optimizer         | Adam (lr = 0.0002, weight_decay = 1e-4)    |
| Epochs            | 20                                         |
| Training Hardware | NVIDIA T4 GPU (Kaggle)                     |

---

## Features

### Driving Intelligence
- Neural network predicts steering, throttle, and brake directly from camera images
- Conditional branches for four navigation commands — Follow, Left, Right, Straight
- Traffic light detection and obedience (Red / Yellow / Green states)
- Real-time obstacle detection with speed-adaptive braking distances
- Route planning using CARLA `GlobalRoutePlanner`

### Control System
- PID-style speed controller (target: 35 km/h, hard cap: 45 km/h)
- Curve detection with automatic speed reduction (15–22 km/h in bends)
- Progressive braking that scales with current vehicle speed
- Lane-change overtaking state machine (Left / Right / Reverse / Teleport fallback)

### Safety & Recovery
- Off-road detection with automatic road recovery behaviour
- Stuck detection with reverse manoeuvre and teleport fallback
- Red light grace period to prevent false overtake triggers after traffic stops
- Collision cooldown mechanism to prevent inflated collision counts

### Visualization & Evaluation
- Real-time dashboard HUD showing speed, steering angle, current status, and distance to destination
- Comprehensive evaluation metrics and safety scoring
- Terminal log output with per-second status (speed, command, obstacle state, traffic light)

### Custom Map Support
- OpenStreetMap → OpenDRIVE converter for importing real-world road networks
- CUSAT campus map integration with landmark-to-spawn-point mapping

---

## Dataset & Training

### Data Collection

- **Simulator:** CARLA 0.9.10, Town01
- **Sessions:** 5 manual driving sessions with varying traffic and weather
- **Total Frames:** 176,256
- **Sensors:** RGB Camera at 800×600, 100° FOV
- **Labels:** Speed, steering, throttle, brake, navigation command per frame
- **Preprocessing:** Images resized from 800×600 → 200×88 for training

### Training Configuration

| Parameter        | Value        |
|------------------|--------------|
| Epochs           | 20           |
| Batch Size       | 128          |
| Learning Rate    | 0.0002       |
| Optimizer        | Adam         |
| Weight Decay     | 1e-4         |
| Loss (Controls)  | MSE          |
| Loss (Speed)     | MSE          |
| GPU              | NVIDIA T4    |
| Platform         | Kaggle       |

### Training Pipeline

1. **Collect** — Drive manually in CARLA using `collect_data.py`; saves images + `measurements.csv` per session
2. **Prepare** — Run `prepare_dataset.py` to resize images and organise sessions for upload
3. **Train** — Train CILRS model on Kaggle using the Jupyter notebook (`notebook.ipynb`)
4. **Deploy** — Load the saved model checkpoint in `autonomous_drive.py` for inference

---

## Project Structure

```
CILRS-Autonomous-Driving-CARLA/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── model/
│   ├── autonomous_drive.py     # Main autonomous driving agent (V6.0)
│   ├── collect_data.py         # Manual data collection script
│   ├── prepare_dataset.py      # Image resizing and dataset preparation
│   ├── osm_to_xodr.py          # OpenStreetMap → OpenDRIVE converter
│   ├── load_cusat.py           # CUSAT campus map loader for CARLA
│   └── map_landmarks.py        # GPS landmark → spawn point mapper
│
├── training/
│   └── notebook.ipynb          # Kaggle training notebook
│
├── demo/
│   ├── result.png              # Training loss curves (all 6 plots)
│   ├── output1.png             # Simulation screenshot — normal driving
│   └── output2.png             # Simulation screenshot — red light stop
│
└── docs/
    └── (architecture diagrams)
```

---

## Installation & Setup

### Prerequisites

- CARLA 0.9.10 Simulator
- Python 3.7
- NVIDIA GPU with CUDA support
- Conda (recommended)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CILRS-Autonomous-Driving-CARLA.git
cd CILRS-Autonomous-Driving-CARLA
```

### Step 2 — Create Conda Environment

```bash
conda create -n carla_project python=3.7 -y
conda activate carla_project
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify CARLA Installation

```bash
cd ~/carla_simulator
./CarlaUE4.sh -opengl -quality-level=Low
```

---

## Usage

### Running Autonomous Driving

**Terminal 1 — Start the CARLA simulator:**
```bash
cd ~/carla_simulator
./CarlaUE4.sh -opengl -quality-level=Low &
```

**Terminal 2 — Launch the autonomous agent:**
```bash
cd model/
python autonomous_drive.py --spawn 0 --destination 50 \
    --vehicles 40 --pedestrians 5 --duration 300
```

### Command Line Arguments

| Argument         | Default | Description                             |
|------------------|---------|-----------------------------------------|
| `--spawn`        | 0       | Spawn point index for the ego vehicle   |
| `--destination`  | None    | Target spawn point index                |
| `--vehicles`     | 15      | Number of NPC vehicles in the scene     |
| `--pedestrians`  | 10      | Number of NPC pedestrians               |
| `--duration`     | 180     | Driving duration in seconds             |
| `--map`          | None    | Use `'cusat'` for the custom campus map |
| `--no-hud`       | False   | Disable the real-time dashboard HUD     |

### Data Collection

```bash
python collect_data.py --duration 300 --vehicles 15
```

### Dataset Preparation

```bash
python prepare_dataset.py
# Resizes images from 800×600 to 200×88
# Output: ~/carla_simulator/training_data/
```

---

## Custom Map — CUSAT Campus

> ⚠️ **Status: Experimental** — The CUSAT custom map pipeline (OSM → OpenDRIVE → CARLA) is implemented and functional for map loading and spawn point generation. However, autonomous driving on the CUSAT map has **not been fully validated** — the trained CILRS model currently produces reliable results only on **CARLA Town01**, which matches the training distribution.
>
> The CUSAT map tools are included as a proof-of-concept for real-world map integration and future work.

This project includes a toolchain for driving on a custom map of the **CUSAT (Cochin University of Science and Technology) campus**, converted from real OpenStreetMap data into CARLA's OpenDRIVE format.

### Workflow

```
OpenStreetMap (.osm)  ──►  osm_to_xodr.py  ──►  cusat_campus.xodr
                                                         │
                                                  load_cusat.py
                                                         │
                                               CARLA World (CUSAT)
                                                         │
                                               map_landmarks.py
                                                         │
                                          Spawn points ↔ Landmarks
```

### Loading the CUSAT Map

```bash
# Step 1: Convert OSM to OpenDRIVE (run once)
python osm_to_xodr.py ~/Downloads/map.osm

# Step 2: Load map into CARLA (keep running, press Ctrl+C after "Map loaded")
python load_cusat.py

# Step 3: Map landmarks to spawn points
python map_landmarks.py

# Step 4: Attempt driving (experimental — results may vary)
python autonomous_drive.py --map cusat --spawn 0 \
    --vehicles 5 --pedestrians 0 --duration 180
```

### Campus Landmarks Mapped

| Landmark                    | Notes                        |
|-----------------------------|------------------------------|
| CUSAT Main Gate             | Primary entry point          |
| School of Engineering       | Main academic block          |
| Dept of Computer Science    | Project home department      |
| Admin Building              | Central administration       |
| Central Library             | Library complex              |
| Seminar Complex             | Conference facilities        |
| University Hostel           | Student accommodation        |
| Sports Arena                | Recreation area              |
| Marine Sciences / Ship Tech | Specialized departments      |
| Boat Jetty / Lakeside       | Waterfront access            |

### Fully Tested Command (Town01)

For reliable autonomous driving results, use Town01 with the command below — this is the configuration used in the final evaluation:

```bash
python autonomous_drive.py --spawn 0 --destination 50 \
    --vehicles 40 --pedestrians 5 --duration 300
```

---

## Tech Stack

| Technology      | Version   | Purpose                          |
|-----------------|-----------|----------------------------------|
| Python          | 3.7       | Primary programming language     |
| PyTorch         | 1.13      | Deep learning framework          |
| CARLA           | 0.9.10    | Autonomous driving simulator     |
| OpenCV          | 4.6       | Image capture and processing     |
| ResNet-34       | —         | Visual feature extraction        |
| NumPy           | —         | Numerical computation            |
| OpenStreetMap   | —         | Real-world map data source       |

---

## References

1. Codevilla, F. et al. — *[Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://arxiv.org/abs/1904.08980)* (CILRS paper), ICCV 2019
2. Codevilla, F. et al. — *End-to-end Driving via Conditional Imitation Learning*, ICRA 2018
3. [CARLA Simulator Documentation](https://carla.readthedocs.io/en/0.9.10/)
4. He, K. et al. — *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016

---

## Author

**Rohith**
B.Tech — Computer Science and Engineering

---

## License

This project was developed as a B.Tech CSE Major Project for academic and educational purposes.
