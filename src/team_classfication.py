import logging
from typing import Iterator, List, Tuple

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from config.config import (
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
)
from src.player_detection import resolve_goalkeepers_team_id
from Team.team import TeamClassifier
from utils.utils import get_crops, validate_video_path

# Configure logger
logger = logging.getLogger(__name__)


class TeamClassificationProcessor:
    """
    Process football videos for team classification.

    This class handles the classification of players into teams based on jersey colors,
    and provides visualization of the results.
    """

    def __init__(
        self,
        model_path: str = PLAYER_DETECTION_MODEL_PATH,
        device: str = "cpu",
        min_consecutive_frames: int = 3,
        max_frames_for_fitting: int = 100,
    ) -> None:
        """
        Initialize the team classification processor.

        Args:
            model_path: Path to the player detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            min_consecutive_frames: Minimum consecutive frames for tracking
            max_frames_for_fitting: Maximum frames to use for team classification fitting
        """
        self.model_path = model_path
        self.device = device
        self.min_consecutive_frames = min_consecutive_frames
        self.max_frames_for_fitting = max_frames_for_fitting

        # Initialize components
        self.model = YOLO(self.model_path).to(device=self.device)
        self.tracker = sv.ByteTrack(
            minimum_consecutive_frames=self.min_consecutive_frames
        )
        self.team_classifier = TeamClassifier(device=self.device)

        self.track_team_votes = {}

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
            source_path=source_video_path, stride=STRIDE
        )

        crops = []
        frame_count = 0

        for frame in tqdm(frame_generator, desc="Collecting crops"):
            # Limit the number of frames processed
            if frame_count >= self.max_frames_for_fitting:
                break

            result = self.model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
            crops += get_crops(frame, player_detections)

            frame_count += 1

            # Limit the number of crops to avoid excessive memory usage
            if len(crops) > 1000:
                logger.info(f"Collected {len(crops)} crops, stopping collection")
                break

        logger.info(f"Collected {len(crops)} player crops from {frame_count} frames")
        return crops

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process a single frame for team classification.

        Args:
            frame: Video frame to process

        Returns:
            Tuple containing annotated frame, merged detections, and labels
        """
        # Detect players
        result = self.model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        # Filter players and classify by team
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = self.team_classifier.predict(crops) if crops else np.array([])

        # Stabilize team labels using tracker IDs
        if len(players_team_id) > 0:
            stable_ids = []
            for tracker_id, team_id in zip(players.tracker_id, players_team_id):
                votes = self.track_team_votes.get(
                    tracker_id,
                    np.zeros(self.team_classifier.n_cluster, dtype=int),
                )
                votes[team_id] += 1
                self.track_team_votes[tracker_id] = votes
                stable_ids.append(int(np.argmax(votes)))
            players_team_id = np.array(stable_ids)

        # Handle goalkeepers
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = (
            resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            if len(goalkeepers) > 0 and len(players_team_id) > 0
            else np.array([])
        )

        # Handle referees
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Merge all detections
        merged_detections = sv.Detections.merge([players, goalkeepers, referees])

        # Create color lookup array
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )

        # Create labels
        labels = [str(tracker_id) for tracker_id in merged_detections.tracker_id]

        # Annotate frame
        annotated_frame = frame.copy()
        if len(merged_detections) > 0 and len(color_lookup) == len(merged_detections):
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, merged_detections, custom_color_lookup=color_lookup
            )
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame,
                merged_detections,
                labels,
                custom_color_lookup=color_lookup,
            )

        return annotated_frame, merged_detections, labels

    def process_video(self, source_video_path: str) -> Iterator[np.ndarray]:
        """
        Process a video for team classification.

        Args:
            source_video_path: Path to the source video file

        Yields:
            Annotated video frames with team classifications

        Raises:
            FileNotFoundError: If the video file doesn't exist
        """
        # Validate input
        if not validate_video_path(source_video_path):
            raise FileNotFoundError(f"Video not found: {source_video_path}")

        # Collect player crops and fit the team classifier
        crops = self.collect_player_crops(source_video_path)

        if not crops:
            logger.warning(
                "No player crops collected, team classification may be inaccurate"
            )
        else:
            logger.info(f"Fitting team classifier with {len(crops)} crops")
            self.team_classifier.fit(crops)

        # Process video frames
        logger.info("Starting main frame processing loop...")
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

        for frame in frame_generator:
            annotated_frame, _, _ = self.process_frame(frame)
            yield annotated_frame


def run_team_classification(
    source_video_path: str, device: str
) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path: Path to the source video
        device: Device to run inference on ('cpu', 'cuda', 'mps')

    Yields:
        Iterator over annotated frames with team classifications
    """
    processor = TeamClassificationProcessor(device=device)
    yield from processor.process_video(source_video_path)