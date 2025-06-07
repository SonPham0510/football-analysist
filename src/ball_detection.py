import logging
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np
import supervision as sv
from ultralytics import YOLO

from Ball.ball import BallAnnotator, BallTracker
from config.config import BALL_DETECTION_MODEL_PATH
from utils.utils import validate_video_path

# Configure logger
logger = logging.getLogger(__name__)


class BallDetector:
    """
    Ball detection and tracking class for football videos.

    This class handles the detection and tracking of the ball in football videos
    using YOLOv8 and custom tracking algorithms.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = BALL_DETECTION_MODEL_PATH,
        device: str = "cpu",
        tracker_buffer_size: int = 20,
        annotator_radius: int = 6,
        annotator_buffer_size: int = 10,
        slice_wh: Tuple[int, int] = (640, 640),
        nms_threshold: float = 0.1,
    ) -> None:
        """
        Initialize the ball detector.

        Args:
            model_path: Path to the YOLOv8 ball detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            tracker_buffer_size: Buffer size for ball trajectory tracking
            annotator_radius: Radius of the ball marker in annotated frames
            annotator_buffer_size: Buffer size for trajectory visualization
            slice_wh: Width and height of the inference slices
            nms_threshold: Non-maximum suppression threshold
        """
        self.model_path = Path(model_path)
        self.device = device
        self.tracker_buffer_size = tracker_buffer_size
        self.annotator_radius = annotator_radius
        self.annotator_buffer_size = annotator_buffer_size
        self.slice_wh = slice_wh
        self.nms_threshold = nms_threshold

        # Initialize components
        self._load_model()
        self.ball_tracker = BallTracker(buffer_size=self.tracker_buffer_size)
        self.ball_annotator = BallAnnotator(
            radius=self.annotator_radius, buffer_size=self.annotator_buffer_size
        )

    def _load_model(self) -> None:
        """Load the YOLOv8 ball detection model."""
        try:
            self.model = YOLO(self.model_path).to(device=self.device)
            logger.info(f"Ball detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ball detection model: {e}")
            raise

    def _inference_callback(self, image_slice: np.ndarray) -> sv.Detections:
        """
        Perform ball detection on an image slice.

        Args:
            image_slice: Image slice to perform detection on

        Returns:
            Detection results
        """
        result = self.model(image_slice, imgsz=self.slice_wh[0], verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for ball detection and tracking.

        Args:
            source_video_path: Path to the source video file

        Yields:
            Annotated video frames with ball detection and tracking

        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        # Validate input
        if not validate_video_path(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")

        # Create frame generator and slicer
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

        slicer = sv.InferenceSlicer(
            callback=self._inference_callback,
            slice_wh=self.slice_wh,
            overlap_ratio_wh=None,
            overlap_wh=(0, 0),
        )

        # Process each frame
        for frame in frame_generator:
            # Detect and track balls
            detections = slicer(frame).with_nms(threshold=self.nms_threshold)
            detections = self.ball_tracker.update(detections)

            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = self.ball_annotator.annotate(annotated_frame, detections)

            yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path: Path to the source video
        device: Device to run the model on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames
    """
    detector = BallDetector(device=device)
    yield from detector.process_video(source_video_path)
