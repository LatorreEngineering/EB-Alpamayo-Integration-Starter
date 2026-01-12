# EB-Alpamayo-Integration-Starter

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NVIDIA Alpamayo](https://img.shields.io/badge/NVIDIA-Alpamayo%201-76B900.svg)](https://huggingface.co/nvidia/Alpamayo-R1-10B)

> **Professional integration toolkit for NVIDIA Alpamayo 1 Vision-Language-Action model in automotive research contexts**

Inspired by Elektrobit's expertise in safe software-defined vehicle (SDV) integration, this repository provides a clean, modular, and production-oriented starting point for researchers and engineers exploring NVIDIA's groundbreaking Alpamayo 1 reasoning VLA model for autonomous driving research.

---

## 🎯 What is This?

A **complete, ready-to-run** development environment that bridges cutting-edge AI research (Alpamayo 1) with automotive software engineering best practices:

- ✅ **Docker-based setup** with CUDA support
- ✅ **Clean Python interfaces** with type hints and comprehensive documentation
- ✅ **Reasoning trace visualization** — see the model's chain-of-thought and chain-of-causation
- ✅ **Trajectory validation** with basic safety plausibility checks
- ✅ **AlpaSim integration** for closed-loop simulation experiments
- ✅ **Modular architecture** ready for AUTOSAR/ROS2 integration
- ✅ **Production-grade code quality** with tests, linting, and structured logging

---

## 🚨 Important Notice

**FOR RESEARCH & EXPERIMENTATION PURPOSES ONLY**

This software is **NOT** intended for production vehicles or safety-critical applications. It does not implement ISO 26262 functional safety requirements, SOTIF validation, or any certification-grade safety mechanisms. Use only in controlled research environments.

---

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** with ≥24GB VRAM (RTX 4090, A5000, or better)
  - For GPUs with less VRAM, see [quantization options](#quantization)
- **Docker** with NVIDIA Container Toolkit
- **Hugging Face account** with access to [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/EB-Alpamayo-Integration-Starter.git
cd EB-Alpamayo-Integration-Starter

# Build Docker image
docker build -t eb-alpamayo:latest .

# Or use docker-compose (includes AlpaSim)
docker-compose up -d

# Login to Hugging Face (required for model download)
huggingface-cli login
```

### Run Your First Inference

```bash
# Start Jupyter Lab
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace eb-alpamayo:latest

# Open notebooks/01_basic_inference.ipynb in your browser
# The notebook will guide you through:
#   1. Loading Alpamayo 1 model
#   2. Running inference on sample driving scenario
#   3. Visualizing predicted trajectory + reasoning traces
```

---

## 📁 Repository Structure

```
.
├── notebooks/              # Interactive Jupyter tutorials
│   ├── 01_basic_inference.ipynb           # Load model, run inference, visualize
│   ├── 02_reasoning_visualization.ipynb   # Parse and analyze reasoning traces
│   └── 03_alpasim_closed_loop_example.ipynb  # Closed-loop simulation demo
├── src/                    # Core Python modules
│   ├── alpamayo_wrapper.py    # Clean model loading/inference interface
│   ├── reasoning_parser.py    # Extract & analyze chain-of-thought traces
│   ├── trajectory_viz.py      # Trajectory visualization utilities
│   └── safety_checks.py       # Basic trajectory plausibility validation
├── examples/               # Sample data and scenarios
│   └── sample_scenario/    # Example input data (or download instructions)
├── tests/                  # Pytest test suite
└── Dockerfile              # Multi-stage CUDA-enabled container
```

---

## 🔬 Key Features

### 1. **Alpamayo 1 Model Wrapper**

Clean, type-safe interface for model inference:

```python
from src.alpamayo_wrapper import AlpamayoInference

# Initialize model with automatic device management
model = AlpamayoInference(
    model_name="nvidia/Alpamayo-R1-10B",
    device="cuda",
    dtype="bfloat16"
)

# Run inference on driving scenario
result = model.predict(
    images=camera_frames,        # List[PIL.Image]
    prompt="Navigate this intersection safely",
    max_reasoning_steps=10
)

# Access structured output
trajectory = result.trajectory    # Planned waypoints
reasoning = result.reasoning_trace  # Chain-of-thought steps
```

### 2. **Reasoning Trace Visualization**

Understand *how* the model makes decisions:

```python
from src.reasoning_parser import ReasoningParser

parser = ReasoningParser()
parsed = parser.extract_reasoning_steps(result.reasoning_trace)

# Analyze decision process
for step in parsed.steps:
    print(f"Step {step.id}: {step.thought}")
    print(f"  Confidence: {step.confidence}")
    print(f"  Safety concerns: {step.safety_flags}")
```

### 3. **Safety Plausibility Checks**

Basic validation inspired by automotive safety principles:

```python
from src.safety_checks import TrajectoryValidator

validator = TrajectoryValidator(
    max_lateral_accel=4.0,  # m/s² (ISO 26262 inspired)
    max_longitudinal_accel=3.0,
    collision_lookahead=3.0  # seconds
)

validation = validator.check(trajectory, scene_context)

if not validation.is_safe:
    print(f"Safety concerns: {validation.violations}")
```

### 4. **AlpaSim Integration**

Closed-loop evaluation in NVIDIA's simulation environment:

```python
from src.alpamayo_wrapper import AlpamayoInference
import alpasim

# Initialize simulator
sim = alpasim.Simulator(scenario="urban_intersection_01")

# Run closed-loop with Alpamayo 1
model = AlpamayoInference("nvidia/Alpamayo-R1-10B")

for timestep in sim.run():
    observation = sim.get_observation()
    action = model.predict(
        images=observation.cameras,
        prompt=f"Current speed: {observation.ego_speed} m/s"
    )
    sim.step(action.trajectory)
```

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_wrapper.py -v
```

---

## 📊 Example Outputs

### Trajectory Visualization
![Trajectory Example](docs/images/trajectory_viz_example.png)
*Bird's-eye view of predicted trajectory with lane boundaries and obstacles*

### Reasoning Trace
![Reasoning Example](docs/images/reasoning_trace_example.png)
*Step-by-step chain-of-thought with confidence scores and safety assessments*

*(Placeholder images — generate during your experiments)*

---

## 🔧 Advanced Usage

### <a name="quantization"></a>Quantization for Smaller GPUs

If you have <24GB VRAM, use 8-bit or 4-bit quantization:

```python
from src.alpamayo_wrapper import AlpamayoInference

model = AlpamayoInference(
    model_name="nvidia/Alpamayo-R1-10B",
    load_in_8bit=True,  # Requires ~14GB VRAM
    # load_in_4bit=True,  # Requires ~8GB VRAM (experimental)
)
```

### Custom Safety Validators

Extend the `TrajectoryValidator` for domain-specific checks:

```python
from src.safety_checks import TrajectoryValidator, SafetyViolation

class HighwayValidator(TrajectoryValidator):
    def check_minimum_speed(self, trajectory):
        """Highway-specific: ensure minimum speed compliance"""
        if trajectory.avg_speed < 65 * 0.447:  # 65 mph in m/s
            return SafetyViolation(
                severity="WARNING",
                message="Below minimum highway speed"
            )
        return None
```

### Integration with ROS2

Example ROS2 node wrapper (see `examples/ros2_integration/` for full code):

```python
import rclpy
from sensor_msgs.msg import Image
from src.alpamayo_wrapper import AlpamayoInference

class AlpamayoPlannerNode(Node):
    def __init__(self):
        super().__init__('alpamayo_planner')
        self.model = AlpamayoInference("nvidia/Alpamayo-R1-10B")
        self.subscription = self.create_subscription(
            Image, '/camera/front', self.camera_callback, 10)
    
    def camera_callback(self, msg):
        # Convert ROS Image → PIL Image → Inference
        result = self.model.predict(images=[self.ros_to_pil(msg)])
        # Publish trajectory to /planned_path
        ...
```

---

## 📚 Documentation & Resources

- **Alpamayo 1 Model Card**: [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- **AlpaSim Simulator**: [github.com/NVlabs/alpasim](https://github.com/NVlabs/alpasim)
- **NVIDIA CES 2026 Announcement**: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- **ISO 26262 Overview**: [ISO Standards Catalog](https://www.iso.org/standard/68383.html)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code style guidelines (Black, Ruff, type hints)
- Automotive-specific documentation requirements
- Safety-aware development practices
- How to submit pull requests

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) file for details.

**Why Apache 2.0?**
- Widely adopted in automotive open-source (AUTOSAR Adaptive, ROS2)
- Provides explicit patent grant (critical for automotive AI/ML)
- Compatible with most commercial automotive software stacks
- Allows derivative works while protecting contributors

---

## 🙏 Acknowledgments

- **NVIDIA** for open-sourcing Alpamayo 1 and AlpaSim
- **Elektrobit (EB)** for inspiration in safe SDV software integration practices
- The broader **autonomous driving research community**

---

## ⚠️ Disclaimer

This is an independent open-source project for research purposes. It is:

- **NOT** officially endorsed by NVIDIA, Elektrobit, or any automotive OEM
- **NOT** validated for functional safety (ISO 26262, SOTIF)
- **NOT** suitable for deployment in real vehicles
- **NOT** a substitute for professional automotive software engineering

Use at your own risk in controlled research environments only.

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-org/EB-Alpamayo-Integration-Starter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/EB-Alpamayo-Integration-Starter/discussions)
- **Contact**: https://github.com/LatorreEngineering

---

