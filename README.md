# Prismata - Multi-Domain Video Event Intelligence

Prismata is a high-performance, domain-agnostic framework for detecting refined events in videos using temporal deep learning (CNN + LSTM). Originally designed for cricket, Prismata now provides a "prism" through which raw video can be refracted into categorizeable, actionable event data across sports, industrial monitoring, and surveillance.

## Features

- **Refracted Observation**: Plugin-style domain architecture that breaks complex video into discrete events.
- **Pre-built Domains**: High-precision configurations for Cricket, Soccer, and Warehouse environments.
- **Extensible Architecture**: Seamlessly add new domains with custom event logic.
- **Temporal Mastery**: Combines ResNet spatial encoding with Bidirectional LSTM temporal modeling.
- **End-to-End Intelligence**: Integrated pipeline for training, high-speed inference, and label management.

## Supported Domains

### Cricket
- **Event Type**: Bowling deliveries
- **Use Case**: Automatically detect and extract delivery clips from cricket match footage
- **Optimized For**: Broadcast quality cricket videos

### Soccer  
- **Event Types**: Goals, penalties, corner kicks, free kicks
- **Use Case**: Detect key moments in soccer matches
- **Optimized For**: Match footage analysis

### Warehouse
- **Event Types**: Package pickups, forklift movements, safety violations
- **Use Case**: Surveillance and operational monitoring
- **Optimized For**: Fixed-camera warehouse surveillance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/event-recording.git
cd event-recording

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For labeling tool
pip install -e ".[labeling]"
```

## Quick Start

### Training a Model

```bash
# Train cricket detector using Prismata
prismata-train --domain cricket \
    --labels-dir data/labels \
    --videos-dir data/raw \
    --epochs 50

# Train with specific domain configuration
prismata-train --domain soccer \
    --config configs/domains/soccer.yaml
```

### Running Detection

```bash
# Detect events in a video stream
prismata-detect video.mp4 \
    --checkpoint models/cricket_detector_best.pt \
    --output results/

# Batch process multiple videos
prismata-detect batch videos/ \
    --checkpoint models/soccer_detector_best.pt \
    --output results/
```

### Labeling Data

```bash
# Launch the Prismata labeling interface
prismata-label --domain cricket
```

## Project Structure

```
event-recording/
├── src/
│   ├── core/                 # Core abstractions
│   │   ├── domain.py        # Domain interface and registry
│   │   ├── base_detector.py # Base model classes
│   │   └── base_dataset.py  # Base dataset classes
│   ├── domains/             # Domain implementations
│   │   ├── cricket/         # Cricket domain
│   │   ├── soccer/          # Soccer domain
│   │   └── warehouse/       # Warehouse domain
│   ├── models/              # Shared model components
│   ├── data/                # Data utilities
│   ├── training/            # Training pipeline
│   └── inference/           # Inference pipeline
├── scripts/                 # CLI scripts
│   ├── train.py            # Training script
│   ├── detect.py           # Detection script
│   └── label.py            # Labeling tool
└── configs/                 # Configuration files
    └── domains/            # Domain-specific configs
        ├── cricket.yaml
        ├── soccer.yaml
        └── warehouse.yaml
```

## Adding a New Domain

1. **Create domain directory**:
```bash
mkdir -p src/domains/your_domain
```

2. **Implement domain class**:
```python
# src/domains/your_domain/domain.py
from src.core.domain import Domain, EventType, register_domain

@register_domain
class YourDomain(Domain):
    @property
    def name(self) -> str:
        return "your_domain"
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.YOUR_EVENT]
    
    # Implement other required methods...
```

3. **Create detector and dataset**:
```python
# src/domains/your_domain/detector.py
from src.core.base_detector import BaseEventDetector

class YourDetector(BaseEventDetector):
    # Implement forward() and predict_proba()
    pass

# src/domains/your_domain/dataset.py
from src.core.base_dataset import BaseEventDataset

class YourDataset(BaseEventDataset):
    # Implement _load_labels() and _create_samples()
    pass
```

4. **Create configuration**:
```yaml
# configs/domains/your_domain.yaml
domain: your_domain
labels_dir: data/your_domain/labels
videos_dir: data/your_domain/raw
# ... other parameters
```

5. **Register domain** (auto-registered via decorator):
```python
# src/domains/__init__.py
from src.domains.your_domain import YourDomain  # noqa: F401
```

## Configuration

Each domain has its own configuration file in `configs/domains/`. Key parameters:

- `domain`: Domain name
- `window_size`: Number of frames per sample
- `target_fps`: Target FPS for frame extraction
- `backbone`: CNN architecture (resnet18, resnet34, resnet50)
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate

See example configs for more details.

## Model Architecture

The default architecture uses:
- **Frame Encoder**: ResNet CNN for spatial features
- **Temporal Model**: Bidirectional LSTM for temporal modeling
- **Classifier**: MLP head for binary classification

Each domain can customize these components as needed.

## License

MIT

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{event_detector,
  title={Event Detector: Multi-Domain Video Event Detection},
  author={Vijay Ponduru},
  year={2026}
}
```
