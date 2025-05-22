
import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np
import pytesseract
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from Team.team import TeamClassifier
from config.config import (
    BOX_ANNOTATOR,
    BOX_LABEL_ANNOTATOR,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    STRIDE
)
from utils.utils import get_crops, validate_video_path
# Configure logger
logger = logging.getLogger(__name__)

# Constants
JERSEY_NUMBER_THRESHOLD = 5  # Number of times the same jersey number must be observed


class JerseyNumberDetector:
    """
    Jersey number detection and recognition for football videos.
    
    This class handles the detection and recognition of jersey numbers
    in football videos, while also classifying players by team.
    """
    
    def __init__(
        self,
        model_path: str = PLAYER_DETECTION_MODEL_PATH,
        device: str = "cpu",
        min_consecutive_frames: int = 3,
        jersey_number_threshold: int = JERSEY_NUMBER_THRESHOLD
    ) -> None:
        """
        Initialize the jersey number detector.
        
        Args:
            model_path: Path to the player detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            min_consecutive_frames: Minimum consecutive frames for tracking
            jersey_number_threshold: Number of times a jersey number must be observed
                                    before being assigned to a player
        """
        self.model_path = model_path
        self.device = device
        self.min_consecutive_frames = min_consecutive_frames
        self.jersey_number_threshold = jersey_number_threshold
        
        # Initialize components
        self._load_model()
        self.tracker = sv.ByteTrack(minimum_consecutive_frames=self.min_consecutive_frames)
        self.team_classifier = TeamClassifier(device=device)
        
        # Initialize jersey number tracking
        self.jersey_numbers_history = defaultdict(lambda: defaultdict(int))
        self.assigned_jersey_numbers = defaultdict(dict)
    
    def _load_model(self) -> None:
        """Load the player detection model."""
        try:
            self.model = YOLO(self.model_path).to(device=self.device)
            logger.info(f"Player detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load player detection model: {e}")
            raise
    
    def collect_player_crops(self, source_video_path: str) -> List[np.ndarray]:
        """
        Collect player crops for team classification fitting.
        
        Args:
            source_video_path: Path to the source video
            
        Returns:
            List of player image crops
        """
        logger.info("Collecting player crops for team classification...")
        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path, 
            stride=STRIDE
        )
        
        crops = []
        for frame in tqdm(frame_generator, desc='Collecting crops'):
            result = self.model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
            crops += get_crops(frame, player_detections)
            
            # Limit the number of crops to avoid excessive memory usage
            if len(crops) > 1000:
                break
        
        logger.info(f"Collected {len(crops)} player crops")
        return crops
    
    def recognize_jersey_number(self, crop: np.ndarray) -> str:
        """
        Recognize jersey number from a player crop using OCR.
        
        Args:
            crop: Cropped image of a player
            
        Returns:
            Recognized jersey number as string
        """
        # Preprocess the crop to improve OCR accuracy
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to isolate the numbers
        _, thresh_crop = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Use OCR to read the jersey number
        jersey_number = pytesseract.image_to_string(
            thresh_crop,
            config='--psm 7 -c tessedit_char_whitelist=0123456789'
        ).strip()
        
        return jersey_number
    
    def update_jersey_tracking(
        self,
        tracker_id: int,
        team_id: int,
        jersey_number: str
    ) -> str:
        """
        Update jersey number tracking and assign numbers to players.
        
        This function:
        1. Updates the history count for this jersey number
        2. Checks if the number has been seen enough times to assign it
        3. Ensures the number isn't already assigned to another player on the same team
        
        Args:
            tracker_id: Player's tracker ID
            team_id: Player's team ID
            jersey_number: Recognized jersey number
            
        Returns:
            The assigned jersey number or empty string
        """
        # Update jersey number history
        counts = self.jersey_numbers_history[tracker_id]
        counts[jersey_number] += 1
        
        # Check if already assigned
        if tracker_id in self.assigned_jersey_numbers[team_id]:
            return self.assigned_jersey_numbers[team_id][tracker_id]
        
        # Check if any jersey number has reached the threshold
        for number, count in counts.items():
            if count >= self.jersey_number_threshold and number.isdigit():
                # Check if the jersey number is already assigned to another player in the same team
                team_numbers = self.assigned_jersey_numbers[team_id]
                if number not in team_numbers.values():
                    self.assigned_jersey_numbers[team_id][tracker_id] = number
                    return number
        
        return ""
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, sv.Detections, List[str]]:
        """
        Process a single frame for jersey number detection.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Tuple containing:
                - annotated frame
                - player detections
                - jersey number labels
        """
        # Detect players
        player_result = self.model(frame, imgsz=1280, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = self.tracker.update_with_detections(player_detections)
        
        # Get player crops
        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, players)
        
        # Classify teams
        players_team_id = self.team_classifier.predict(player_crops)
        
        # Process jersey numbers
        labels = []
        for tracker_id, team_id, crop in zip(players.tracker_id, players_team_id, player_crops):
            # Recognize jersey number
            jersey_number = self.recognize_jersey_number(crop)
            
            # Update tracking and get assigned number
            assigned_number = self.update_jersey_tracking(tracker_id, team_id, jersey_number)
            labels.append(assigned_number)
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, players)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, players, labels=labels
        )
        
        return annotated_frame, players, labels
    
    def get_jersey_assignments(self) -> Dict[int, Dict[int, str]]:
        """
        Get the current jersey number assignments.
        
        Returns:
            Dictionary mapping team IDs to dictionaries mapping tracker IDs to jersey numbers
        """
        return dict(self.assigned_jersey_numbers)
    
    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for jersey number detection.
        
        Args:
            source_video_path: Path to the source video file
            
        Yields:
            Annotated video frames with detected jersey numbers
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        # Validate input
        if not validate_video_path(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")
        
        # Collect player crops and fit the team classifier
        crops = self.collect_player_crops(source_video_path)
        
        if not crops:
            logger.warning("No player crops collected, team classification may be inaccurate")
        else:
            logger.info(f"Fitting team classifier with {len(crops)} crops")
            self.team_classifier.fit(crops)
        
        # Reset jersey number tracking
        self.jersey_numbers_history = defaultdict(lambda: defaultdict(int))
        self.assigned_jersey_numbers = defaultdict(dict)
        
        # Process video frames
        logger.info("Starting main frame processing loop...")
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
        
        for frame in frame_generator:
            annotated_frame, _, _ = self.process_frame(frame)
            yield annotated_frame


def run_jersey_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run jersey number detection on a video and yield annotated frames.

    Args:
        source_video_path: Path to the source video
        device: Device to run the model on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames with jersey numbers
    """
    detector = JerseyNumberDetector(device=device)
    yield from detector.process_video(source_video_path)