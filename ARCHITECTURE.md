# Prismata Architecture

## What It Is

Prismata is a multi-domain video event detection framework. It uses CNN+LSTM temporal deep learning to detect events in video — originally cricket bowling deliveries, now extensible to soccer, warehouse surveillance, factory monitoring, etc.

The system covers the full pipeline: labeling, training, inference, and deployment.

---

## Architecture Journey

### Phase 1: Local Prototype (`a7a3d36` → `0c04472`)

Started as a single-domain cricket delivery detector with hardcoded local paths.

- **Model**: ResNet18 (frame encoder) + Bidirectional LSTM (temporal model) + MLP classifier
- **Input**: Video windows `(B, T=8, C=3, H=224, W=224)` → binary logits `(B, 1)`
- **Data**: Local `data/raw/` videos, `data/labels/` JSON files
- **UI**: Streamlit labeler with frame-by-frame annotation, detection, and training tabs
- **Inference**: PyTorch `EventPredictor` with sliding window, threshold-based event grouping

Refactored to multi-domain with plugin architecture:
- `Domain` base class + `@register_domain` decorator
- `DomainRegistry` for dynamic lookup
- Domain-specific models, datasets, and configs

### Phase 2: Cloud Infrastructure (`980e6fa` → `a5204cc`)

Added AWS deployment via Terraform and Docker.

- **Terraform modules**: VPC, S3, RDS (PostgreSQL), EC2, ECS, Lambda orchestrator
- **Docker**: Two-stage build, CPU-only PyTorch (fits 20GB disk), Streamlit server
- **Database**: SQLAlchemy models for videos, labels, models, detections, jobs
- **Minimal environment**: t3.micro → t3.small EC2, single S3 bucket, Cloudflare quick tunnel

### Phase 3: Storage Abstraction (`6ad71d3`)

Wired the labeler to an existing storage abstraction layer, eliminating all hardcoded paths.

```
StorageBackend (abstract)
├── LocalStorageBackend  — filesystem, zero overhead
└── S3StorageBackend     — boto3, local caching, presigned URLs
```

**Key design**: `get_storage_backend()` reads `PRISMATA_STORAGE_BACKEND` env var. All video/label/model/detection operations go through the backend. S3 backend caches downloads locally with MD5-based keys. Video playback uses presigned URLs (no download needed).

Before:
```python
cap = cv2.VideoCapture("data/raw/video.mp4")
labels = json.load(open("data/labels/video.json"))
```

After:
```python
storage = get_storage_backend()
cap = cv2.VideoCapture(str(storage.read_video("video.mp4")))
labels = storage.read_labels("video.json")
```

### Phase 4: Memory Survival (`3b325cf` → `75052d9`)

Deployed to EC2 t3.small (2GB RAM). PyTorch inference consumed ~1.6GB, OOM-killing the Streamlit container and crashing the instance.

**Fixes applied iteratively**:

1. **UI guard** — Disable "Run Detection" button while running to prevent double-clicks
2. **num_workers=0** — Avoid DataLoader worker process memory duplication
3. **Subprocess isolation** — Run detection in `subprocess.run()` so OOM kills the child, not Streamlit
4. **Memory env vars** — `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in Dockerfile
5. **2GB swap file** on EC2 as safety net

Result: Streamlit survives detection failures, but PyTorch still OOMs on 2GB.

### Phase 5: ONNX Runtime (`d2775ec` → `0ecaad7`)

Replaced PyTorch inference with ONNX Runtime to eliminate ~500MB runtime overhead.

```
PyTorch path:  ~1.6GB RSS (torch + model + data)
ONNX path:     ~1.0GB RSS (onnxruntime + model + data)
```

**Components**:
- `onnx_export.py` — One-time conversion: `.pt` checkpoint → `.onnx` (168MB → 56MB)
- `onnx_predictor.py` — Full inference pipeline using only `onnxruntime`, `numpy`, `cv2`. No PyTorch import at all.
- Labeler auto-detects `.onnx` model in storage and uses it; falls back to PyTorch

**Key constraint**: The ONNX predictor replicates the windowing, normalization, batch inference, and event grouping logic from `EventPredictor` without importing torch. This is intentional — the whole point is avoiding PyTorch's memory footprint.

**Result**: Detection completes on t3.small (2GB) with ~1GB memory, zero swap pressure.

### Phase 6: UX Polish (`98e1f49` → `fcd5fbe`)

- Video selection persists after detection completes (no re-selecting from dropdown)
- Detection results show which runtime was used (ONNX vs PyTorch)
- "Export ONNX" button in training tab for each model

---

## Current Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Labeling │  │ Analysis │  │   Training    │  │
│  │  Tab     │  │   Tab    │  │     Tab       │  │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │              │                │          │
│       │         ┌────▼─────┐    ┌─────▼──────┐  │
│       │         │Subprocess│    │  Trainer    │  │
│       │         │(isolated)│    │ (in-proc)   │  │
│       │         └────┬─────┘    └─────┬──────┘  │
└───────┼──────────────┼────────────────┼──────────┘
        │              │                │
   ┌────▼──────────────▼────────────────▼────┐
   │          Storage Backend                 │
   │  ┌─────────────┐  ┌──────────────────┐  │
   │  │    Local     │  │       S3         │  │
   │  │  Filesystem  │  │  + Local Cache   │  │
   │  └─────────────┘  └──────────────────┘  │
   └─────────────────────────────────────────┘

Detection Subprocess:
   ┌──────────────────────────────────────┐
   │  Prefers ONNX Runtime (if .onnx)     │
   │  Falls back to PyTorch (if .pt only) │
   │  Writes JSON result to temp file     │
   │  Parent reads result after exit      │
   └──────────────────────────────────────┘
```

### Model Pipeline

```
Video (S3/local)
  → Sliding Windows (window_size=8, stride=4-8, target_fps=5-10)
    → Frame Extraction (cv2/Decord, resize to 224x224)
      → Normalize (ImageNet mean/std)
        → Batch Inference (ONNX or PyTorch)
          → Sigmoid → Threshold
            → Event Grouping (min_gap, buffer_seconds)
              → DetectionResult JSON
```

### Domain Plugin System

```
@register_domain
class CricketDomain(Domain):
    def create_model(config) → DeliveryDetector
    def create_dataset(labels, videos) → CricketDataset
    def create_inference_dataset(video) → CricketInferenceDataset

DomainRegistry.get("cricket")  # Returns instance
```

### Key Files

| File | Purpose |
|------|---------|
| `src/data/labeler.py` | Streamlit UI (~2000 lines) — labeling, analysis, training |
| `src/models/detector.py` | CNN+LSTM model architecture |
| `src/inference/predictor.py` | PyTorch inference pipeline |
| `src/inference/onnx_predictor.py` | ONNX inference (no torch) |
| `src/inference/onnx_export.py` | PyTorch → ONNX conversion |
| `src/storage/base.py` | StorageBackend abstract interface |
| `src/storage/s3.py` | S3 backend with caching + presigned URLs |
| `src/training/trainer.py` | Training loop with early stopping |
| `src/core/domain.py` | Domain registry + plugin system |
| `src/core/base_dataset.py` | Base dataset with Decord/OpenCV |

---

## Deployment

**Current**: EC2 t3.small (2 vCPU, 2GB RAM) running Docker container with Cloudflare quick tunnel.

| Component | Where | Notes |
|-----------|-------|-------|
| Streamlit UI | EC2 t3.small | Docker, port 8501 |
| Videos/Labels/Models | S3 | Single bucket with prefixes |
| Detection inference | Subprocess on same EC2 | ONNX Runtime, ~1GB peak |
| Public access | Cloudflare quick tunnel | URL changes on restart |

**Performance** (ONNX, t3.small, batch_size=4, target_fps=5, stride=8):
- ~15-20 min per 10-min video clip
- ~1GB peak memory, no swap
- GPU instance (g4dn.xlarge) would bring this under 1 minute

---

## What's Next

- **GPU inference**: Spin up g4dn.xlarge spot instance on-demand for <1 min detection
- **Streaming results**: Show events in UI as they're detected (not just after completion)
- **Persistent tunnel**: Replace Cloudflare quick tunnel with named tunnel or ALB
- **Batch detection**: Queue multiple videos for sequential processing
