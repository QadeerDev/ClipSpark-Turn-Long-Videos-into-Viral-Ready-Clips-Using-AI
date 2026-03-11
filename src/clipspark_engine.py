"""
ClipSpark AI Engine - Production Pipeline
==========================================

Industrial-grade video processing pipeline for automatic clip generation.
Detects high-interest moments using motion + human presence analysis.

Author: ClipSpark Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the ClipSpark pipeline."""
    
    # Paths (Docker-ready: uses APP_DIR env var)
    base_dir: Path = field(default_factory=lambda: Path(os.getenv("APP_DIR", ".")))
    
    # Video validation
    valid_extensions: tuple = (".mp4", ".mov")
    min_duration: int = 60   # seconds
    max_duration: int = 300  # seconds
    
    # Proxy generation
    proxy_height: int = 360
    proxy_fps: int = 10
    
    # Scene detection
    scene_threshold: float = 27.0
    min_scene_length: float = 1.0  # seconds
    
    # Signal weights (0.6 Motion / 0.4 Human for "Spark" branding)
    motion_weight: float = 0.6
    human_weight: float = 0.4
    
    # Clip selection
    target_durations: tuple = (15, 30, 60)
    duration_tolerance: float = 0.2  # 20%
    
    # Human detection
    yolo_sample_interval: int = 5
    yolo_confidence: float = 0.3
    
    # Output
    ffmpeg_preset: str = "medium"
    ffmpeg_crf: int = 23
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output"
    
    @property
    def clips_dir(self) -> Path:
        return self.output_dir / "clips"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and optional file output."""
    
    logger = logging.getLogger("clipspark")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Custom formatter with phase indicators
    class PhaseFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m'
        }
        
        def format(self, record):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            # Add phase prefix if available
            phase = getattr(record, 'phase', None)
            if phase:
                record.msg = f"[{phase}] {record.msg}"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            return f"{color}{timestamp} | {record.levelname:8} | {record.msg}{reset}"
    
    console_handler.setFormatter(PhaseFormatter())
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VideoMetadata:
    """Video file metadata."""
    filepath: str
    duration: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    filesize_mb: float


@dataclass
class ValidationResult:
    """Video validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[VideoMetadata] = None


@dataclass
class Scene:
    """Detected scene in video."""
    id: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionSignal:
    """Motion analysis results."""
    raw_scores: np.ndarray
    smoothed_scores: np.ndarray
    fps: float
    duration: float


@dataclass
class HumanSignal:
    """Human detection results."""
    raw_scores: np.ndarray
    smoothed_scores: np.ndarray
    person_counts: np.ndarray
    fps: float
    duration: float


@dataclass
class FusedSignal:
    """Fused motion + human signal."""
    per_second_scores: np.ndarray
    motion_weight: float
    human_weight: float
    duration: float


@dataclass
class ClipCandidate:
    """Selected clip candidate."""
    target_duration: int
    start_time: float
    end_time: float
    actual_duration: float
    score: float
    source_scene_ids: List[int]
    selection_tier: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedClip:
    """Final extracted clip."""
    filename: str
    filepath: str
    target_duration: int
    expected_duration: float
    actual_duration: float
    start_time: float
    end_time: float
    score: float
    filesize_mb: float
    passed_validation: bool
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# PHASE 1: INGESTION & OPTIMIZATION
# ============================================================================

class IngestionPhase:
    """Phase 1: Video ingestion and proxy generation."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using OpenCV."""
        path = Path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            duration = frame_count / fps if fps > 0 else 0
            filesize_mb = path.stat().st_size / (1024 * 1024)
            
            return VideoMetadata(
                filepath=str(path.absolute()),
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                codec=codec,
                filesize_mb=filesize_mb
            )
        finally:
            cap.release()
    
    def validate_video(self, video_path: str) -> ValidationResult:
        """Validate video against ClipSpark requirements."""
        errors = []
        warnings = []
        path = Path(video_path)
        
        self.logger.info(f"Validating: {path.name}", extra={"phase": "PHASE 1"})
        
        if not path.exists():
            return ValidationResult(False, [f"File not found: {video_path}"])
        
        if path.suffix.lower() not in self.config.valid_extensions:
            errors.append(f"Invalid format: {path.suffix}. Accepted: {self.config.valid_extensions}")
        
        try:
            metadata = self.get_video_metadata(video_path)
        except Exception as e:
            return ValidationResult(False, [f"Cannot read video: {e}"])
        
        if metadata.duration < self.config.min_duration:
            errors.append(f"Too short: {metadata.duration:.1f}s < {self.config.min_duration}s")
        elif metadata.duration > self.config.max_duration:
            errors.append(f"Too long: {metadata.duration:.1f}s > {self.config.max_duration}s")
        
        if metadata.height > 1080:
            warnings.append(f"Resolution {metadata.width}x{metadata.height} exceeds 1080p")
        
        if errors:
            for e in errors:
                self.logger.error(f"Validation error: {e}", extra={"phase": "PHASE 1"})
        else:
            self.logger.info(
                f"Valid: {metadata.duration:.1f}s, {metadata.width}x{metadata.height}, {metadata.filesize_mb:.1f}MB",
                extra={"phase": "PHASE 1"}
            )
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def generate_proxy(self, input_path: str, output_dir: str) -> str:
        """Generate low-resolution proxy video."""
        inp = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        proxy_name = f"{inp.stem}_proxy_{self.config.proxy_height}p_{self.config.proxy_fps}fps.mp4"
        output_path = out_dir / proxy_name
        
        self.logger.info(f"Generating proxy: {proxy_name}", extra={"phase": "PHASE 1"})
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(inp),
            "-vf", f"scale=-2:{self.config.proxy_height}",
            "-r", str(self.config.proxy_fps),
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")
        
        if not output_path.exists():
            raise RuntimeError(f"Proxy file not created: {output_path}")
        
        proxy_size = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"Proxy generated: {proxy_size:.1f}MB", extra={"phase": "PHASE 1"})
        
        return str(output_path)


# ============================================================================
# PHASE 2: SCENE DETECTION
# ============================================================================

class SceneDetectionPhase:
    """Phase 2: Scene boundary detection."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def detect_scenes(self, video_path: str) -> List[Scene]:
        """Detect scene boundaries using PySceneDetect."""
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        
        self.logger.info(f"Detecting scenes with threshold {self.config.scene_threshold}", extra={"phase": "PHASE 2"})
        
        video = open_video(video_path)
        scene_manager = SceneManager()
        
        min_scene_frames = int(self.config.min_scene_length * video.frame_rate)
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.config.scene_threshold,
                min_scene_len=min_scene_frames
            )
        )
        
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scenes.append(Scene(
                id=i + 1,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                duration=end.get_seconds() - start.get_seconds(),
                start_frame=start.get_frames(),
                end_frame=end.get_frames()
            ))
        
        self.logger.info(f"Detected {len(scenes)} scenes", extra={"phase": "PHASE 2"})
        
        if scenes:
            avg_duration = sum(s.duration for s in scenes) / len(scenes)
            self.logger.debug(f"Avg scene duration: {avg_duration:.1f}s", extra={"phase": "PHASE 2"})
        
        return scenes


# ============================================================================
# PHASE 3: INTELLIGENCE EXTRACTION
# ============================================================================

class IntelligenceExtractionPhase:
    """Phase 3: Motion and human detection."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._yolo_model = None
    
    @property
    def yolo_model(self):
        """Lazy load YOLO model."""
        if self._yolo_model is None:
            from ultralytics import YOLO
            self.logger.info("Loading YOLOv8n model...", extra={"phase": "PHASE 3"})
            # Look for model in models/ directory or current directory
            model_paths = [
                Path("models/yolov8n.pt"),
                Path("/app/models/yolov8n.pt"),
                Path("yolov8n.pt")
            ]
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = str(path)
                    break
            if model_path is None:
                model_path = "yolov8n.pt"  # Let ultralytics download it
            self._yolo_model = YOLO(model_path)
            self.logger.info("YOLOv8n model loaded", extra={"phase": "PHASE 3"})
        return self._yolo_model
    
    def analyze_motion(self, video_path: str, smoothing_window: float = 1.0) -> MotionSignal:
        """Analyze motion using frame difference algorithm."""
        self.logger.info("Analyzing motion (frame difference)...", extra={"phase": "PHASE 3"})
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        prev_gray = cv2.GaussianBlur(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            (21, 21), 0
        )
        
        raw_scores = []
        
        for _ in tqdm(range(total_frames - 1), desc="Motion Analysis", unit="frame"):
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.GaussianBlur(
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
                (21, 21), 0
            )
            
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            raw_scores.append(np.sum(frame_diff))
            prev_gray = curr_gray
        
        cap.release()
        
        # Normalize
        raw_scores = np.array(raw_scores, dtype=np.float32)
        if raw_scores.max() > raw_scores.min():
            raw_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        
        # Smooth
        kernel_size = max(1, int(smoothing_window * fps))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(raw_scores, kernel, mode='same')
        
        # Resample to per-second
        fps_int = int(fps)
        num_seconds = int(np.ceil(len(smoothed) / fps_int))
        per_second = np.array([
            smoothed[i * fps_int:(i + 1) * fps_int].mean()
            for i in range(num_seconds)
        ])
        
        self.logger.info(
            f"Motion: {len(per_second)}s, range {per_second.min():.2f}-{per_second.max():.2f}",
            extra={"phase": "PHASE 3"}
        )
        
        return MotionSignal(raw_scores, per_second, fps, total_frames / fps)
    
    def analyze_humans(self, video_path: str) -> HumanSignal:
        """Analyze human presence using YOLOv8."""
        self.logger.info(
            f"Analyzing humans (YOLOv8, interval={self.config.yolo_sample_interval})...",
            extra={"phase": "PHASE 3"}
        )
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = frame_width * frame_height
        
        raw_scores = []
        person_counts = []
        frame_indices = []
        
        sample_range = range(0, total_frames, self.config.yolo_sample_interval)
        
        for fidx in tqdm(sample_range, desc="Human Detection", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.yolo_model(frame, verbose=False, conf=self.config.yolo_confidence)
            boxes = results[0].boxes
            person_boxes = boxes.xyxy[boxes.cls == 0].cpu().numpy()
            
            num_people = len(person_boxes)
            total_area = sum(
                (b[2] - b[0]) * (b[3] - b[1])
                for b in person_boxes
            ) / frame_area if frame_area > 0 else 0
            
            # Human score: Count × (1 + Prominence)
            score = num_people * (1 + total_area)
            
            raw_scores.append(score)
            person_counts.append(num_people)
            frame_indices.append(fidx)
        
        cap.release()
        
        # Normalize
        raw_scores = np.array(raw_scores, dtype=np.float32)
        if raw_scores.max() > raw_scores.min():
            raw_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        
        # Map to per-second
        duration = total_frames / fps
        num_seconds = int(np.ceil(duration))
        per_second = np.zeros(num_seconds)
        per_count = np.zeros(num_seconds)
        
        for i, fidx in enumerate(frame_indices):
            sec = int(fidx / fps)
            if sec < num_seconds:
                per_second[sec] = max(per_second[sec], raw_scores[i])
                per_count[sec] = max(per_count[sec], person_counts[i])
        
        self.logger.info(
            f"Humans: {len(per_second)}s, max people: {int(per_count.max())}",
            extra={"phase": "PHASE 3"}
        )
        
        return HumanSignal(raw_scores, per_second, per_count.astype(np.int32), fps, duration)


# ============================================================================
# PHASE 4: SCORING & FUSION
# ============================================================================

class ScoringPhase:
    """Phase 4: Signal fusion and scene scoring."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def fuse_signals(self, motion: MotionSignal, human: HumanSignal) -> FusedSignal:
        """Fuse motion and human signals."""
        motion_weight = self.config.motion_weight
        human_weight = self.config.human_weight
        
        self.logger.info(
            f"Fusing signals (motion={motion_weight}, human={human_weight})",
            extra={"phase": "PHASE 4"}
        )
        
        # Align lengths
        min_len = min(len(motion.smoothed_scores), len(human.smoothed_scores))
        m_scores = motion.smoothed_scores[:min_len]
        h_scores = human.smoothed_scores[:min_len]
        
        # Handle no-human edge case
        if h_scores.max() == 0:
            self.logger.warning("No humans detected - using 100% motion", extra={"phase": "PHASE 4"})
            motion_weight = 1.0
            human_weight = 0.0
        
        # Weighted fusion
        fused = motion_weight * m_scores + human_weight * h_scores
        
        # Normalize
        if fused.max() > fused.min():
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        
        self.logger.info(
            f"Fused: {len(fused)}s, range {fused.min():.2f}-{fused.max():.2f}, mean={fused.mean():.2f}",
            extra={"phase": "PHASE 4"}
        )
        
        return FusedSignal(fused, motion_weight, human_weight, min_len)
    
    def score_scenes(self, scenes: List[Scene], fused: FusedSignal) -> List[Scene]:
        """Calculate score for each scene."""
        self.logger.info(f"Scoring {len(scenes)} scenes...", extra={"phase": "PHASE 4"})
        
        scores = fused.per_second_scores
        
        for scene in scenes:
            start_sec = int(scene.start_time)
            end_sec = min(int(scene.end_time), len(scores))
            
            if start_sec < end_sec and start_sec < len(scores):
                scene.score = float(np.mean(scores[start_sec:end_sec]))
            else:
                scene.score = 0.0
        
        # Log top scenes
        sorted_scenes = sorted(scenes, key=lambda s: s.score, reverse=True)
        self.logger.info("Top 3 scenes by score:", extra={"phase": "PHASE 4"})
        for s in sorted_scenes[:3]:
            self.logger.info(
                f"  Scene {s.id}: {s.start_time:.1f}s-{s.end_time:.1f}s → {s.score:.3f}",
                extra={"phase": "PHASE 4"}
            )
        
        return scenes


# ============================================================================
# PHASE 5: SELECTION
# ============================================================================

class SelectionPhase:
    """Phase 5: Clip candidate selection."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def select_clips(self, scenes: List[Scene], fused: FusedSignal) -> List[ClipCandidate]:
        """Select best clips using refined greedy strategy."""
        self.logger.info(
            f"Selecting clips for durations: {self.config.target_durations}",
            extra={"phase": "PHASE 5"}
        )
        
        sorted_scenes = sorted(scenes, key=lambda s: s.score, reverse=True)
        locked_ids = set()
        candidates = []
        scores = fused.per_second_scores
        video_duration = fused.duration
        
        for target in self.config.target_durations:
            self.logger.debug(f"Finding best {target}s clip...", extra={"phase": "PHASE 5"})
            
            if video_duration < target:
                self.logger.warning(f"Video too short for {target}s clip", extra={"phase": "PHASE 5"})
                continue
            
            min_dur = target * (1 - self.config.duration_tolerance)
            max_dur = target * (1 + self.config.duration_tolerance)
            
            best_candidate = None
            
            for scene in sorted_scenes:
                if scene.id in locked_ids:
                    continue
                
                # TIER 1: Perfect fit
                if min_dur <= scene.duration <= max_dur:
                    best_candidate = ClipCandidate(
                        target_duration=target,
                        start_time=scene.start_time,
                        end_time=scene.end_time,
                        actual_duration=scene.duration,
                        score=scene.score,
                        source_scene_ids=[scene.id],
                        selection_tier=1
                    )
                    self.logger.info(
                        f"  {target}s: Tier 1 - Scene {scene.id} ({scene.duration:.1f}s)",
                        extra={"phase": "PHASE 5"}
                    )
                    break
                
                # TIER 2: Merge neighbors
                if scene.duration < min_dur:
                    scene_idx = next((i for i, s in enumerate(scenes) if s.id == scene.id), None)
                    if scene_idx is not None:
                        merged_duration = scene.duration
                        merged_ids = [scene.id]
                        merged_end = scene.end_time
                        
                        for next_scene in scenes[scene_idx + 1:]:
                            if next_scene.id in locked_ids:
                                continue
                            merged_duration += next_scene.duration
                            merged_ids.append(next_scene.id)
                            merged_end = next_scene.end_time
                            if merged_duration >= min_dur:
                                break
                        
                        if min_dur <= merged_duration <= max_dur:
                            start_sec = int(scene.start_time)
                            end_sec = min(int(merged_end), len(scores))
                            merged_score = np.mean(scores[start_sec:end_sec]) if end_sec > start_sec else 0
                            
                            best_candidate = ClipCandidate(
                                target_duration=target,
                                start_time=scene.start_time,
                                end_time=merged_end,
                                actual_duration=merged_duration,
                                score=merged_score,
                                source_scene_ids=merged_ids,
                                selection_tier=2
                            )
                            self.logger.info(
                                f"  {target}s: Tier 2 - Merged {len(merged_ids)} scenes ({merged_duration:.1f}s)",
                                extra={"phase": "PHASE 5"}
                            )
                            break
                
                # TIER 3: Sub-segment extraction
                if scene.duration > max_dur:
                    start_sec = int(scene.start_time)
                    end_sec = int(scene.end_time)
                    
                    best_subscore = -1
                    best_start = start_sec
                    
                    for win_start in range(start_sec, end_sec - target + 1):
                        win_end = min(win_start + target, len(scores))
                        if win_end <= win_start:
                            continue
                        win_score = np.mean(scores[win_start:win_end])
                        if win_score > best_subscore:
                            best_subscore = win_score
                            best_start = win_start
                    
                    if best_subscore >= 0:
                        best_candidate = ClipCandidate(
                            target_duration=target,
                            start_time=float(best_start),
                            end_time=float(best_start + target),
                            actual_duration=float(target),
                            score=best_subscore,
                            source_scene_ids=[scene.id],
                            selection_tier=3
                        )
                        self.logger.info(
                            f"  {target}s: Tier 3 - Sub-segment at {best_start}s",
                            extra={"phase": "PHASE 5"}
                        )
                        break
            
            if best_candidate:
                for sid in best_candidate.source_scene_ids:
                    locked_ids.add(sid)
                candidates.append(best_candidate)
            else:
                self.logger.warning(f"  {target}s: No suitable clip found", extra={"phase": "PHASE 5"})
        
        self.logger.info(f"Selected {len(candidates)} clips", extra={"phase": "PHASE 5"})
        return candidates


# ============================================================================
# PHASE 6: PRODUCTION
# ============================================================================

class ProductionPhase:
    """Phase 6: Clip extraction and quality validation."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Quick metadata extraction for validation."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return VideoMetadata(
            filepath=video_path,
            duration=frame_count / fps if fps > 0 else 0,
            width=0, height=0, fps=fps,
            frame_count=frame_count, codec="",
            filesize_mb=Path(video_path).stat().st_size / (1024 * 1024)
        )
    
    def extract_clips(
        self,
        candidates: List[ClipCandidate],
        source_video: str,
        output_dir: str
    ) -> List[ExtractedClip]:
        """Extract clips from source video using FFmpeg."""
        self.logger.info(f"Extracting {len(candidates)} clips...", extra={"phase": "PHASE 6"})
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted = []
        
        for i, candidate in enumerate(candidates):
            clip_name = f"clip_{i+1:02d}_{candidate.target_duration}s.mp4"
            clip_path = output_path / clip_name
            
            self.logger.info(
                f"  [{i+1}/{len(candidates)}] {clip_name}: "
                f"{candidate.start_time:.1f}s-{candidate.end_time:.1f}s ({candidate.actual_duration:.1f}s)",
                extra={"phase": "PHASE 6"}
            )
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(candidate.start_time),
                "-i", source_video,
                "-t", str(candidate.actual_duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", self.config.ffmpeg_preset,
                "-crf", str(self.config.ffmpeg_crf),
                str(clip_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            errors = []
            passed = True
            actual_duration = 0.0
            filesize_mb = 0.0
            
            if result.returncode != 0:
                errors.append(f"FFmpeg failed: {result.stderr[:200]}")
                passed = False
            elif not clip_path.exists():
                errors.append("Output file not created")
                passed = False
            else:
                filesize_mb = clip_path.stat().st_size / (1024 * 1024)
                
                if filesize_mb < 0.001:
                    errors.append(f"File too small: {filesize_mb:.3f}MB")
                    passed = False
                
                try:
                    clip_meta = self._get_video_metadata(str(clip_path))
                    actual_duration = clip_meta.duration
                    
                    # FIX #1: Compare to expected, not target
                    dur_diff = abs(actual_duration - candidate.actual_duration)
                    if dur_diff > 0.5:
                        errors.append(
                            f"FFmpeg glitch: Output {actual_duration:.1f}s != "
                            f"Expected {candidate.actual_duration:.1f}s"
                        )
                        passed = False
                except Exception as e:
                    errors.append(f"Cannot read metadata: {e}")
                    passed = False
            
            status = "✓" if passed else "✗"
            self.logger.info(
                f"    {status} {actual_duration:.1f}s, {filesize_mb:.2f}MB",
                extra={"phase": "PHASE 6"}
            )
            
            if errors:
                for err in errors:
                    self.logger.warning(f"      {err}", extra={"phase": "PHASE 6"})
            
            extracted.append(ExtractedClip(
                filename=clip_name,
                filepath=str(clip_path),
                target_duration=candidate.target_duration,
                expected_duration=candidate.actual_duration,
                actual_duration=actual_duration,
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                score=candidate.score,
                filesize_mb=filesize_mb,
                passed_validation=passed,
                validation_errors=errors
            ))
        
        passed_count = sum(1 for c in extracted if c.passed_validation)
        self.logger.info(
            f"Extraction complete: {passed_count}/{len(extracted)} passed validation",
            extra={"phase": "PHASE 6"}
        )
        
        return extracted
    
    def generate_output(self, clips: List[ExtractedClip], output_dir: str) -> Dict[str, Any]:
        """Generate final JSON output."""
        output = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_clips": len(clips),
            "clips": []
        }
        
        for clip in clips:
            output["clips"].append({
                "filename": clip.filename,
                "duration": round(clip.actual_duration, 2),
                "target_duration": clip.target_duration,
                "expected_duration": round(clip.expected_duration, 2),
                "start_time": round(clip.start_time, 2),
                "end_time": round(clip.end_time, 2),
                "score": round(clip.score, 3),
                "filesize_mb": round(clip.filesize_mb, 2),
                "download_url": f"/clips/{clip.filename}",
                "passed_validation": clip.passed_validation
            })
        
        output_path = Path(output_dir) / "result.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Output saved: {output_path}", extra={"phase": "PHASE 6"})
        
        return output


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ClipSparkPipeline:
    """Main ClipSpark processing pipeline."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize phases
        self.phase1 = IngestionPhase(config, logger)
        self.phase2 = SceneDetectionPhase(config, logger)
        self.phase3 = IntelligenceExtractionPhase(config, logger)
        self.phase4 = ScoringPhase(config, logger)
        self.phase5 = SelectionPhase(config, logger)
        self.phase6 = ProductionPhase(config, logger)
    
    def run(self, video_path: str) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("🎬 CLIPSPARK ENGINE - STARTING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Input: {video_path}")
        self.logger.info(f"Config: Motion={self.config.motion_weight}, Human={self.config.human_weight}")
        self.logger.info("")
        
        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.clips_dir.mkdir(parents=True, exist_ok=True)
        
        # PHASE 1: Ingestion
        self.logger.info("─" * 40)
        validation = self.phase1.validate_video(video_path)
        if not validation.is_valid:
            raise ValueError(f"Validation failed: {validation.errors}")
        
        proxy_path = self.phase1.generate_proxy(video_path, str(self.config.output_dir))
        self.logger.info("")
        
        # PHASE 2: Scene Detection
        self.logger.info("─" * 40)
        scenes = self.phase2.detect_scenes(proxy_path)
        self.logger.info("")
        
        # PHASE 3: Intelligence Extraction
        self.logger.info("─" * 40)
        motion_signal = self.phase3.analyze_motion(proxy_path)
        human_signal = self.phase3.analyze_humans(proxy_path)
        self.logger.info("")
        
        # PHASE 4: Scoring
        self.logger.info("─" * 40)
        fused_signal = self.phase4.fuse_signals(motion_signal, human_signal)
        scenes = self.phase4.score_scenes(scenes, fused_signal)
        self.logger.info("")
        
        # PHASE 5: Selection
        self.logger.info("─" * 40)
        candidates = self.phase5.select_clips(scenes, fused_signal)
        self.logger.info("")
        
        # PHASE 6: Production
        self.logger.info("─" * 40)
        extracted = self.phase6.extract_clips(
            candidates,
            video_path,  # Use original video, not proxy
            str(self.config.clips_dir)
        )
        output = self.phase6.generate_output(extracted, str(self.config.output_dir))
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("🎉 PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Time elapsed: {elapsed:.1f}s")
        self.logger.info(f"Clips generated: {len(extracted)}")
        self.logger.info(f"Output: {self.config.clips_dir}")
        
        return output


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="ClipSpark AI Engine - Automatic video clip generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (MP4/MOV, 1-5 minutes)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.6,
        help="Motion signal weight (default: 0.6)"
    )
    
    parser.add_argument(
        "--human-weight",
        type=float,
        default=0.4,
        help="Human signal weight (default: 0.4)"
    )
    
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[15, 30, 60],
        help="Target clip durations in seconds (default: 15 30 60)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Create config
    config = PipelineConfig(
        motion_weight=args.motion_weight,
        human_weight=args.human_weight,
        target_durations=tuple(args.durations)
    )
    
    if args.output:
        config.base_dir = Path(args.output).parent
    
    # Run pipeline
    try:
        # Check FFmpeg first
        if sys.platform == 'win32':
             try:
                 import winreg
                 with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment') as key:
                     system_path, _ = winreg.QueryValueEx(key, 'Path')
                 with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment') as key:
                     user_path, _ = winreg.QueryValueEx(key, 'Path')
                 os.environ['PATH'] = f"{system_path};{user_path}"
             except Exception as e:
                 logger.warning(f"Could not refresh PATH from registry: {e}")
        
        # Verify FFmpeg is reachable
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg check passed", extra={"phase": "STARTUP"})
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found! Please install it or check your PATH.")
            logger.error("Install via: winget install ffmpeg")
            sys.exit(1)

        pipeline = ClipSparkPipeline(config, logger)
        result = pipeline.run(args.video)
        
        print(f"\n✅ Success! Generated {len(result['clips'])} clips")
        print(f"   Output: {config.clips_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
