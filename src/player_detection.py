import logging
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config.config import (
    BOX_ANNOTATOR,
    BOX_LABEL_ANNOTATOR,
    PLAYER_DETECTION_MODEL_PATH,
)
from utils.utils import validate_video_path

# Configure logger
logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    Player detection class for football videos.
    
    This class handles the detection of players, goalkeepers, and referees
    in football videos using YOLOv8.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = PLAYER_DETECTION_MODEL_PATH,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
        image_size: int = 1280,
    ) -> None:
        """
        Initialize the player detector.
        
        Args:
            model_path: Path to the YOLOv8 player detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence_threshold: Confidence threshold for detections
            image_size: Image size for inference
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        
        # Initialize model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the YOLOv8 player detection model."""
        try:
            self.model = YOLO(self.model_path).to(device=self.device)
            logger.info(f"Player detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load player detection model: {e}")
            raise
    
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players in a frame.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Detected players
        """
        result = self.model(
            frame, 
            imgsz=self.image_size, 
            conf=self.confidence_threshold,
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(result)
        return detections
    
    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for player detection.
        
        Args:
            source_video_path: Path to the source video file
            
        Yields:
            Annotated video frames with detected players
            
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
            detections = self.detect_players(frame)
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = BOX_ANNOTATOR.annotate(
                annotated_frame, detections
            )
            annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections
            )
            
            yield annotated_frame


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on proximity to team centroids.

    Args:
        players: Detections of all players
        players_team_id: Array containing team IDs of detected players
        goalkeepers: Detections of goalkeepers

    Returns:
        Array containing team IDs for the detected goalkeepers
    """
    # Get goalkeeper coordinates (bottom center)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )
    
    # Get player coordinates (bottom center)
    players_xy = players.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )
    
    # Calculate team centroids
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    
    # Assign goalkeepers to teams based on proximity
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        
    return np.array(goalkeepers_team_id)


def run_player_detection(
    source_video_path: str, 
    device: str
) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path: Path to the source video
        device: Device to run the model on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames with player detections
    """
    detector = PlayerDetector(device=device)
    yield from detector.process_video(source_video_path)