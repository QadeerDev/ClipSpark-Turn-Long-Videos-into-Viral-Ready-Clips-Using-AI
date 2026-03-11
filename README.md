# 🎬 ClipSpark

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Celery](https://img.shields.io/badge/Celery-Redis-green.svg)

**ClipSpark** is an industrial-grade AI video processing pipeline that automatically detects and extracts the most engaging moments from long-form video content. Powered by **YOLOv8 human detection** and **motion analysis**, ClipSpark intelligently scores every second of your footage and produces ready-to-share clips in 15s, 30s, and 60s formats — fully automated, zero manual editing required.

---

## ✨ Features

### 🎯 **Core Intelligence**
- **AI-Powered Moment Detection**: Fuses motion intensity and human presence signals to score every second of video
- **YOLOv8 Human Analysis**: Detects people, counts them, and measures their screen prominence
- **Scene Boundary Detection**: Uses PySceneDetect to identify natural cut points before clip selection
- **Weighted Signal Fusion**: Configurable 60% motion / 40% human weighting (the "Spark" formula)

### 🚀 **Production Pipeline (6 Phases)**
- **Phase 1 – Ingestion**: Video validation + low-resolution proxy generation via FFmpeg
- **Phase 2 – Scene Detection**: Content-aware scene boundary detection
- **Phase 3 – Intelligence Extraction**: Frame-level motion analysis + YOLOv8 human detection
- **Phase 4 – Scoring & Fusion**: Per-second signal fusion and scene scoring
- **Phase 5 – Selection**: Tiered clip candidate selection (perfect fit → merge → sub-segment)
- **Phase 6 – Production**: High-quality FFmpeg extraction + JSON result output

### 🌐 **API & Worker Architecture**
- **FastAPI REST API**: Upload videos, poll job status, and download clips via HTTP
- **Celery + Redis Queue**: Asynchronous background processing with job tracking
- **Docker-Ready**: Environment variables and volume mounts preconfigured for containers
- **UUID Job IDs**: Collision-free job and filename management

### 📊 **Output & Configuration**
- **Multi-Duration Clips**: Automatically generates 15s, 30s, and 60s clips per video
- **JSON Result Manifest**: Structured metadata including score, timestamps, filesize, and download URLs
- **Validation Pipeline**: Each extracted clip is verified against expected duration and file integrity
- **Flexible CLI**: Full command-line interface with configurable weights, durations, and log levels

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9+**
- **FFmpeg** installed and on PATH
- **Redis** server (for async API mode)
- **CUDA GPU** (optional, recommended for faster YOLO inference)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/ClipSpark.git
cd ClipSpark

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r config/requirements.txt

# Verify FFmpeg is available
ffmpeg -version
```

### Docker Install

```bash
# Build and run with Docker Compose
docker compose up --build

# Or manually
docker build -t clipspark .
docker run -e APP_DIR=/app -v $(pwd)/output:/app/output clipspark
```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Basic usage — generates 15s, 30s, 60s clips
python src/clipspark_engine.py your_video.mp4

# Custom output directory
python src/clipspark_engine.py your_video.mp4 -o ./my_clips

# Custom signal weights
python src/clipspark_engine.py your_video.mp4 --motion-weight 0.7 --human-weight 0.3

# Custom target clip durations
python src/clipspark_engine.py your_video.mp4 --durations 10 30 45 90

# Debug mode with log file
python src/clipspark_engine.py your_video.mp4 --log-level DEBUG --log-file pipeline.log
```

### API Usage

```bash
# Start Redis
redis-server

# Start Celery worker
celery -A src.clipspark_worker worker --loglevel=info

# Start FastAPI server
uvicorn src.clipspark_api:app --host 0.0.0.0 --port 8000
```

**Upload a video:**
```bash
curl -X POST "http://localhost:8000/process-video" \
     -F "file=@your_video.mp4"
# Response: {"job_id": "abc-123", "status": "queued"}
```

**Check job status:**
```bash
curl "http://localhost:8000/job-status/abc-123"
# Response: {"job_id": "abc-123", "status": "processing"}
```

**Get result:**
```bash
curl "http://localhost:8000/job-result/abc-123"
```

**Download a clip:**
```bash
curl -O "http://localhost:8000/clips/clip_01_15s.mp4"
```

---

## ⚙️ Configuration Options

### CLI Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video` | str | **Required** | Input video path (MP4/MOV, 1–5 minutes) |
| `-o` / `--output` | str | `./output` | Output directory for clips and manifest |
| `--motion-weight` | float | `0.6` | Weight of motion signal in fusion score |
| `--human-weight` | float | `0.4` | Weight of human presence signal in fusion score |
| `--durations` | int+ | `15 30 60` | Target clip durations in seconds |
| `--log-level` | str | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--log-file` | str | None | Optional file path for log output |

### Pipeline Configuration (`PipelineConfig`)

| Parameter | Default | Description |
|---|---|---|
| `valid_extensions` | `.mp4`, `.mov` | Accepted input formats |
| `min_duration` | `60s` | Minimum input video length |
| `max_duration` | `300s` | Maximum input video length |
| `proxy_height` | `360px` | Proxy video resolution for analysis |
| `proxy_fps` | `10` | Proxy video framerate |
| `scene_threshold` | `27.0` | PySceneDetect content sensitivity |
| `yolo_confidence` | `0.3` | YOLO person detection confidence threshold |
| `yolo_sample_interval` | `5` | Analyze every Nth frame for humans |
| `duration_tolerance` | `20%` | Acceptable deviation from target clip duration |
| `ffmpeg_preset` | `medium` | FFmpeg encoding speed/quality preset |
| `ffmpeg_crf` | `23` | FFmpeg constant rate factor (lower = higher quality) |

---

## 📋 Output Structure

```
output/
├── {stem}_proxy_360p_10fps.mp4     # Low-res analysis proxy
├── clips/
│   ├── clip_01_15s.mp4             # Best 15-second clip
│   ├── clip_02_30s.mp4             # Best 30-second clip
│   └── clip_03_60s.mp4             # Best 60-second clip
└── result.json                     # Full processing manifest
```

### `result.json` Format

```json
{
  "status": "completed",
  "timestamp": "2025-09-08T14:32:00",
  "total_clips": 3,
  "clips": [
    {
      "filename": "clip_01_15s.mp4",
      "duration": 15.04,
      "target_duration": 15,
      "start_time": 42.0,
      "end_time": 57.0,
      "score": 0.847,
      "filesize_mb": 12.3,
      "download_url": "/clips/clip_01_15s.mp4",
      "passed_validation": true
    }
  ]
}
```

---

## 🧠 How ClipSpark Scores Video

ClipSpark uses a two-signal fusion model:

```
Final Score(t) = 0.6 × Motion(t) + 0.4 × Human(t)
```

**Motion Signal**: Computed via frame differencing — absolute pixel difference between consecutive frames, normalized and smoothed per second.

**Human Signal**: YOLOv8 detects people in sampled frames. Each second's score is `num_people × (1 + screen_coverage)`, rewarding both crowd size and prominence.

**Clip Selection** follows a 3-tier strategy per target duration:
1. **Tier 1** — Single scene that naturally fits the duration window
2. **Tier 2** — Merge adjacent scenes to reach the target length
3. **Tier 3** — Slide a window over a long scene to find the highest-scoring sub-segment

---

## 🔧 Troubleshooting

**FFmpeg not found**
```bash
# Linux/macOS
sudo apt install ffmpeg   # or brew install ffmpeg

# Windows
winget install ffmpeg
```

**YOLO model download fails**
```bash
# Pre-download manually into models/
mkdir models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

**Video validation errors**
```
Too short: video must be at least 60s
Too long: video must be at most 300s
Invalid format: only .mp4 and .mov are accepted
```

**No clips selected**
- The video may be too short relative to requested durations
- Try smaller `--durations` values or lower `scene_threshold` in `PipelineConfig`

**Redis connection error (API mode)**
```bash
# Ensure Redis is running
redis-cli ping   # Should return PONG
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

Please include tests where applicable and update this README if you change any configuration options or pipeline behavior.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com/)** — YOLOv8 object detection framework
- **[PySceneDetect](https://scenedetect.com/)** — Scene boundary detection
- **[FFmpeg](https://ffmpeg.org/)** — Video processing and encoding
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern async API framework
- **[Celery](https://docs.celeryq.dev/)** — Distributed task queue
- **OpenCV** — Computer vision and motion analysis

---

<div align="center">

**Made with ❤️ by the ClipSpark Team**

[⭐ Star this repo](https://github.com/your-username/ClipSpark) • [🐛 Report Bug](https://github.com/your-username/ClipSpark/issues) • [💡 Request Feature](https://github.com/your-username/ClipSpark/issues/new)

</div>
