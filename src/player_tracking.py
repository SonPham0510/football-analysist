import logging
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config.config import (
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    PLAYER_DETECTION_MODEL_PATH
)
from utils.utils import validate_video_path

# Configure logger
logger = logging.getLogger(__name__)


class PlayerTracker:
    """
    Player tracking class for football videos.
    
    This class handles tracking of players across frames
    using a YOLOv8-based detector and ByteTrack.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = PLAYER_DETECTION_MODEL_PATH,
        device: str = "cpu",
        image_size: int = 1280,
        min_consecutive_frames: int = 3
    ) -> None:
        """
        Initialize the player tracker.
        
        Args:
            model_path: Path to the YOLOv8 player detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            image_size: Image size for inference
            min_consecutive_frames: Minimum consecutive detections for tracking
        """
        self.model_path = Path(model_path)
        self.device = device
        self.image_size = image_size
        self.min_consecutive_frames = min_consecutive_frames
        
        # Initialize components
        self._load_model()
        self.tracker = sv.ByteTrack(
            minimum_consecutive_frames=self.min_consecutive_frames
        )
        
    def _load_model(self) -> None:
        """Load the YOLOv8 player detection model."""
        try:
            self.model = YOLO(self.model_path).to(device=self.device)
            logger.info(f"Player detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load player detection model: {e}")
            raise
    
    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for player tracking.
        
        Args:
            source_video_path: Path to the source video file
            
        Yields:
            Annotated video frames with tracked players
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        # Validate input
        if not validate_video_path(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")
            
        # Create frame generator
        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path
        )
        
        # Process each frame
        for frame in frame_generator:
            # Detect players
            result = self.model(frame, imgsz=self.image_size, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Track players
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Create labels with tracker IDs
            labels = [str(tracker_id) for tracker_id in tracked_detections.tracker_id]
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, tracked_detections
            )
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, tracked_detections, labels=labels
            )
            
            yield annotated_frame


def run_player_tracking(
    source_video_path: str, 
    device: str
) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames.

    Args:
        source_video_path: Path to the source video
        device: Device to run the model on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames with tracked players
    """
    tracker = PlayerTracker(device=device)
    yield from tracker.process_video(source_video_path)