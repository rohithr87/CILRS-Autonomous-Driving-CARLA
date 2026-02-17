# 🚗 CILRS Autonomous Driving in CARLA Simulator

An end-to-end autonomous driving system using **Conditional Imitation Learning with a ResNet-34 backbone (CILRS)** in the CARLA 0.9.10 simulator. Built as a B.Tech CSE Major Project.

---

## 🎯 Results

| Metric | Value |
|---|---|
| Safety Score | 99.8 / 100 |
| Overall Grade | A+ (Excellent) |
| Total Collisions | 0 |
| Red Light Violations | 0 |
| Steering Correlation | 0.9861 |
| Max Speed | 35.4 km/h |
| Off-road | 0.4% |

> Tested with 40 NPC vehicles + 5 pedestrians for 450 seconds in Town01.

---

## 🏗️ Architecture

```
Camera Image (200×88) → ResNet-34 → Visual Features
                                          ↓
Speed (normalized) → Speed Encoder → Combined Features
                                          ↓
Navigation Command → Branch Selector → [Follow | Left | Right | Straight]
                                          ↓
                                  Predicted Controls
                              [Steering, Throttle, Brake]
                                          ↓
                          Speed Prediction (auxiliary task)
```

### Model Details
- **Backbone:** ResNet-34 (pretrained on ImageNet)
- **Parameters:** 22.4 Million
- **Branches:** 4 conditional branches (Follow, Left, Right, Straight)
- **Training Data:** 176,256 frames across 5 driving sessions
- **Training:** 20 epochs on NVIDIA T4 GPU
- **Validation Loss:** 0.0538
- **Optimizer:** Adam (lr=0.0002, weight_decay=1e-4)

---

## 🔧 Features

### Driving Intelligence
- 🧠 Neural network predicts steering, throttle, and brake from camera images
- 🚦 Traffic light detection and obedience (Red/Yellow/Green)
- 🚗 Real-time obstacle detection with speed-adaptive braking distances
- 🔄 Lane-change overtaking state machine (Left/Right/Reverse/Teleport)
- 📍 Route planning using CARLA GlobalRoutePlanner

### Control System
- ⚡ PID-like speed controller (target: 35 km/h, max: 45 km/h)
- 🔀 Curve detection with automatic speed reduction (15-22 km/h)
- 🛑 Progressive braking (scales with vehicle speed)
- 🏎️ Hard speed cap with progressive braking (45/50/55 km/h)

### Safety & Recovery
- 🛡️ Off-road detection with automatic road recovery
- 🔙 Stuck detection with reverse + teleport fallback
- ⏱️ Red light grace period (prevents false overtake after traffic stops)
- 📊 Collision cooldown (prevents inflated collision counts)

### Visualization
- 📺 Real-time dashboard HUD with speed, steering, status
- 📈 Comprehensive evaluation metrics and scoring

### Custom Map Support
- 🗺️ OpenStreetMap to OpenDRIVE converter
- 🏫 CUSAT campus map integration
- 📌 Landmark-to-spawn-point mapping

---

## 🚀 Quick Start

### Prerequisites
- CARLA 0.9.10 Simulator
- Python 3.7
- NVIDIA GPU with CUDA support

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/CILRS-Autonomous-Driving-CARLA.git
cd CILRS-Autonomous-Driving-CARLA

# Create conda environment
conda create -n carla_project python=3.7 -y
conda activate carla_project

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Terminal 1: Start CARLA simulator
cd ~/carla_simulator && ./CarlaUE4.sh -opengl -quality-level=Low &

# Terminal 2: Run autonomous driving
cd model/
python autonomous_drive.py --spawn 0 --destination 50 \
    --vehicles 40 --pedestrians 5 --duration 300
```

### Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--spawn` | 0 | Spawn point index |
| `--destination` | None | Destination spawn point |
| `--vehicles` | 15 | Number of NPC vehicles |
| `--pedestrians` | 10 | Number of pedestrians |
| `--duration` | 180 | Driving duration (seconds) |
| `--map` | None | Use `'cusat'` for custom map |
| `--no-hud` | False | Disable dashboard HUD |

### Custom Map (CUSAT Campus)

```bash
# Load CUSAT map first
python load_cusat.py  # Ctrl+C after "Map loaded"

# Then drive
python autonomous_drive.py --map cusat --spawn 0 \
    --vehicles 5 --pedestrians 0 --duration 180
```

---

## 📊 Training Details

### Data Collection
- Manual driving in CARLA Town01
- 5 sessions with varying traffic and weather
- Captured: RGB camera (800×600), speed, controls, navigation commands
- Resized to 200×88 for training

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 128 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Weight Decay | 1e-4 |
| Loss | MSE (controls) + MSE (speed) |
| GPU | NVIDIA T4 (Kaggle) |

### Training Results

| Metric | Value |
|---|---|
| Validation Loss | 0.0538 |
| Steering Correlation | 0.9861 |
| Throttle Correlation | 0.9589 |
| Brake Correlation | 0.9815 |
| Speed Correlation | 0.9844 |

---

## 📁 Project Structure

```
CILRS-Autonomous-Driving-CARLA/
├── README.md
├── requirements.txt
├── .gitignore
├── model/
│   ├── autonomous_drive.py     # Main driving system (V6.0)
│   ├── collect_data.py         # Data collection script
│   ├── prepare_dataset.py      # Data preprocessing/resizing
│   ├── osm_to_xodr.py         # OSM to OpenDRIVE converter
│   ├── load_cusat.py           # Custom map loader
│   └── map_landmarks.py        # Landmark mapping
├── docs/
│   └── (architecture diagrams)
└── demo/
    └── (demo videos)
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.7 | Programming language |
| PyTorch 1.13 | Deep learning framework |
| CARLA 0.9.10 | Driving simulator |
| OpenCV 4.6 | Image processing |
| ResNet-34 | Feature extraction backbone |
| NumPy | Numerical computing |

---

## 📝 References

- CILRS Paper: [Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://arxiv.org/abs/1904.08980)
- [CARLA Simulator](https://carla.org/)
- Conditional Imitation Learning (Codevilla et al., 2018)

---

## 👤 Author

**Rohith** — B.Tech Computer Science and Engineering

---

## 📄 License

This project is for educational purposes as part of a B.Tech CSE Major Project.
