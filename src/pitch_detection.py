import logging
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config.config import CONFIG, PITCH_DETECTION_MODEL_PATH, VERTEX_LABEL_ANNOTATOR
from utils.utils import validate_video_path

# Configure logger
logger = logging.getLogger(__name__)


class PitchDetector:
    """
    Pitch detection class for football videos.

    This class handles the detection of pitch landmarks in football videos
    using a YOLOv8-based model trained for keypoint detection.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = PITCH_DETECTION_MODEL_PATH,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the pitch detector.

        Args:
            model_path: Path to the YOLOv8 pitch detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence_threshold: Confidence threshold for keypoint detection
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 pitch detection model."""
        try:
            self.model = YOLO(self.model_path).to(device=self.device)
            logger.info(f"Pitch detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load pitch detection model: {e}")
            raise

    def detect_keypoints(self, frame: np.ndarray) -> sv.KeyPoints:
        """
        Detect pitch keypoints in a frame.

        Args:
            frame: Video frame to process

        Returns:
            Detected keypoints
        """
        result = self.model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        return keypoints

    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for pitch detection and annotation.

        Args:
            source_video_path: Path to the source video file

        Yields:
            Annotated video frames with pitch keypoints

        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        # Validate input
        if not validate_video_path(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")

        # Create frame generator
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

        # Process each frame
        for frame in frame_generator:
            # Detect keypoints
            keypoints = self.detect_keypoints(frame)

            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
                annotated_frame, keypoints, CONFIG.labels
            )

            yield annotated_frame


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path: Path to the source video
        device: Device to run the model on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames with pitch keypoints
    """
    detector = PitchDetector(device=device)
    yield from detector.process_video(source_video_path)
