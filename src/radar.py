import json
import logging
from collections import defaultdict

from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from Ball.ball import BallTracker
from config.config import (
    BALL_COLOR_ID,
    BALL_DETECTION_MODEL_PATH,
    CONFIG,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
)
from utils.utils import (
    create_radar_frame,
    euclidean_distance,
    get_crops,
    convert_numpy_types,
)
from src.player_detection import resolve_goalkeepers_team_id
from src.speed import estimate_speed
from Team.team import TeamClassifier

from ViewTransform.view_tranform import ViewTransformer
#from src.jersey_number import JerseyNumberDetector

# Configure logger
logger = logging.getLogger(__name__)

# Constants for team identification
TEAM_A_ID = 0
TEAM_B_ID = 1
#JERSEY_NUMBER_THRESHOLD = 3  # Number of times the same jersey number must be observed


class RadarView:
    """
    Radar view generator for football matches.

    This class generates a top-down schematic view of the football match,
    showing player positions, team affiliations, tracking_id, and ball location.
    """

    def __init__(
        self,
        player_model_path: str,
        pitch_model_path: str,
        ball_model_path: str,
        device: str = "cpu",
        min_consecutive_frames: int = 3,
        ball_buffer_size: int = 20,
        frames_to_fit: int = 50,
    ) -> None:
        """
        Initialize the radar view generator.

        Args:
            player_model_path: Path to the player detection model
            pitch_model_path: Path to the pitch detection model
            ball_model_path: Path to the ball detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            min_consecutive_frames: Minimum consecutive frames for tracking
            ball_buffer_size: Buffer size for ball tracking
            frames_to_fit: Number of frames to use for team classification fitting
        """
        self.player_model_path = player_model_path
        self.pitch_model_path = pitch_model_path
        self.ball_model_path = ball_model_path
        self.device = device
        self.min_consecutive_frames = min_consecutive_frames
        self.ball_buffer_size = ball_buffer_size
        self.frames_to_fit = frames_to_fit

        # Tracking and statistics
        # self.jersey_numbers_history = defaultdict(lambda: defaultdict(int))
        # self.assigned_jersey_numbers = defaultdict(dict)
        self.position_history = defaultdict(list)
        self.possession_counts = {TEAM_A_ID: 0, TEAM_B_ID: 0}
        self.total_frames = 0
        self.all_frames = []  # For JSON export

        # Ball tracking
        self.last_ball_detections = None
        self.ball_missing_frames = 0
        self.ball_missing_threshold = 15

        # Initialize models and components
        self._load_models()
        self._initialize_components()

    def _load_models(self) -> None:
        """Load detection models."""
        logger.info("Loading models...")
        self.player_model = YOLO(self.player_model_path).to(device=self.device)
        self.pitch_model = YOLO(self.pitch_model_path).to(device=self.device)
        self.ball_model = YOLO(self.ball_model_path).to(device=self.device)

    def _initialize_components(self) -> None:
        """Initialize trackers and other components."""
        self.player_tracker = sv.ByteTrack(
            minimum_consecutive_frames=self.min_consecutive_frames
        )
        self.ball_tracker = BallTracker(buffer_size=self.ball_buffer_size)
        self.team_classifier = TeamClassifier(device=self.device)
        # self.jersey_detector = JerseyNumberDetector(
        #     model_path=self.player_model_path,
        #     device=self.device,
        #     jersey_number_threshold=JERSEY_NUMBER_THRESHOLD,
        #     use_gpu=self.device,
        # )

    def fit_team_classifier(self, source_video_path: str) -> None:
        """
        Fit the team classifier using player crops from the video.

        Args:
            source_video_path: Path to the source video
        """
        logger.info("Collecting player crops for team classification...")
        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path, stride=STRIDE
        )

        crops = []
        for frame in tqdm(frame_generator, desc="Collecting crops"):
            result = self.player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            crops += get_crops(
                frame, detections[detections.class_id == PLAYER_CLASS_ID]
            )

            # Limit number of crops to avoid excessive memory usage
            if len(crops) > 1000:
                break

        logger.info(f"Fitting team classifier with {len(crops)} crops")
        self.team_classifier.fit(crops)

    def _detect_pitch_keypoints(
        self, frame: np.ndarray
    ) -> Tuple[sv.KeyPoints, ViewTransformer]:
        """
        Detect pitch keypoints and create a view transformer.

        Args:
            frame: Current video frame

        Returns:
            Tuple containing keypoints and view transformer
        """
        result = self.pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Create a ViewTransformer using the keypoints
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        )

        if mask.sum() < 4:
            logger.warning("Not enough keypoints detected, skipping frame.")
            return None, None
        

        return keypoints, transformer

    def _detect_and_track_players(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> Tuple[sv.Detections, np.ndarray, List[str]]:
        """
        Detect and track players, assigning team IDs

        Args:
            frame: Current video frame

        Returns:
            Tuple containing:
                - player detections
                - team IDs
            
        """

        # Filter out players
        players = detections[detections.class_id == PLAYER_CLASS_ID]

        # Classify players by team
        crops = get_crops(frame, players)
        players_team_id = self.team_classifier.predict(crops)

       
        return players, players_team_id

    def _detect_goalkeepers_and_referees(
        self,
        detections: sv.Detections,
        players: sv.Detections,
        players_team_id: np.ndarray,
    ) -> Tuple[sv.Detections, sv.Detections, np.ndarray]:
        """
        Detect and classify goalkeepers and referees.

        Args:
            detections: All detections
            players: Player detections
            players_team_id: Team IDs for players

        Returns:
            Tuple containing:
                - goalkeeper detections
                - referee detections
                - goalkeeper team IDs
        """
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Resolve goalkeeper team IDs
        goalkeepers_team_id = np.array([], dtype=int)
        if len(goalkeepers) > 0 and len(players) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers
            )

        return goalkeepers, referees, goalkeepers_team_id

    def _detect_and_track_ball(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect and track the ball.

        Args:
            frame: Current video frame

        Returns:
            Ball detections
        """
        ball_result = self.ball_model(frame, imgsz=640, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_result)

        # Filter for ball class
        ball_class_id = 0
        ball_detections = ball_detections[ball_detections.class_id == ball_class_id]
        ball_detections = self.ball_tracker.update(ball_detections)

        # Handle ball disappearance
        if len(ball_detections) == 0:
            if (
                self.last_ball_detections is not None
                and self.ball_missing_frames < self.ball_missing_threshold
            ):
                ball_detections = self.last_ball_detections
                self.ball_missing_frames += 1
            else:
                self.last_ball_detections = None
                self.ball_missing_frames = 0
        else:
            self.last_ball_detections = ball_detections
            self.ball_missing_frames = 0

        # Ensure ball has tracker_id
        if ball_detections.tracker_id is None:
            ball_detections.tracker_id = np.arange(len(ball_detections))

        return ball_detections

    def _calculate_player_speeds(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        transformed_positions: np.ndarray,
        frame_rate: float = 30.0,
    ) -> None:
        """
        Calculate and annotate player speeds.

        Args:
            frame: Current video frame for annotation
            players: Player detections
            transformed_positions: Transformed player positions
            frame_rate: Video frame rate
        """
        for tracker_id, position, bbox in zip(
            players.tracker_id, transformed_positions, players.xyxy
        ):
            # Update position history
            self.position_history[tracker_id].append(position)
            if len(self.position_history[tracker_id]) > 10:  # Keep last 10 positions
                self.position_history[tracker_id].pop(0)

            # Calculate speed
            speed = estimate_speed(self.position_history[tracker_id], frame_rate)

            # Annotate speed on frame
            if tracker_id is not None:
                x, y = int(bbox[0]), int(bbox[1])
                cv2.putText(
                    frame,
                    f"{speed:.2f} m/s",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

    def _determine_ball_possession(
        self,
        frame: np.ndarray,
        transformed_ball_positions: np.ndarray,
        transformed_players_positions: np.ndarray,
        players_team_id: np.ndarray,
    ) -> Optional[int]:
        """
        Determine which team has possession of the ball.

        Args:
            frame: Current video frame for annotation
            transformed_ball_positions: Transformed ball positions
            transformed_players_positions: Transformed player positions
            players_team_id: Team IDs for players

        Returns:
            Team ID with possession
        """
        current_ball_possession_team = None

        if (
            len(transformed_ball_positions) > 0
            and len(transformed_players_positions) > 0
        ):
            ball_pos = transformed_ball_positions[0]
            min_dist = float("inf")
            closest_player_team_id: Optional[int] = None

            # Find the closest player in pitch space
            for pos, team_id in zip(transformed_players_positions, players_team_id):
                dist = euclidean_distance(ball_pos, pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_player_team_id = int(team_id)


            # Update possession counter
            if (
                closest_player_team_id is not None
                and closest_player_team_id in self.possession_counts
            ):
                self.possession_counts[closest_player_team_id] += 1
                current_ball_possession_team = closest_player_team_id

        # Annotate possession info
        self._annotate_possession_info(frame, current_ball_possession_team)

        return current_ball_possession_team

    def _annotate_possession_info(
        self, frame: np.ndarray, current_ball_possession_team: Optional[int]
    ) -> None:
        """
        Annotate possession information on the frame.

        Args:
            frame: Frame to annotate (modified in-place)
            current_ball_possession_team: Team ID with possession
        """
        # Team possession text
        if current_ball_possession_team == TEAM_A_ID:
            cv2.putText(
                frame,
                "team A has it",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        elif current_ball_possession_team == TEAM_B_ID:
            cv2.putText(
                frame,
                "team B has it",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

        # Calculate and display possession percentages
        if self.total_frames > 0:
            pos_a_percent = (
                self.possession_counts[TEAM_A_ID] / self.total_frames
            ) * 100
            pos_b_percent = (
                self.possession_counts[TEAM_B_ID] / self.total_frames
            ) * 100

            cv2.putText(
                frame,
                f"Team A: {pos_a_percent:.1f}% - Team B: {pos_b_percent:.1f}%",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
    

    def _save_frame_data(
        self,
        frame_index: int,
        players: sv.Detections,
        goalkeepers: sv.Detections,
        players_team_id: np.ndarray,
        goalkeepers_team_id: np.ndarray,
        transformed_players_positions: np.ndarray,
        transformed_goalkeepers_positions: np.ndarray,
        transformed_referees_positions: np.ndarray,
        transformed_ball_positions: np.ndarray,
    ) -> None:
        """
        Save frame data for JSON export.

        Args:
            frame_index: Current frame index
            players: Player detections
            goalkeepers: Goalkeeper detections
            players_team_id: Team IDs for players
            goalkeepers_team_id: Team IDs for goalkeepers
            transformed_players_positions: Transformed player positions
            transformed_goalkeepers_positions: Transformed goalkeeper positions
            transformed_referees_positions: Transformed referee positions
            transformed_ball_positions: Transformed ball positions
        """
        frame_data = {
            "frame_index": frame_index,
            "players": [
                {
                    "id": int(tracker_id),
                    "team_id": int(team_id),
                    "position": list(pos),
                   
                }
                for tracker_id, team_id, pos in zip(
                    players.tracker_id, players_team_id, transformed_players_positions
                )
            ],
            "goalkeepers": [
                {
                    "id": int(tracker_id),
                    "team_id": int(team_id),
                    "position": list(pos),
                    
                }
                for tracker_id, team_id, pos in zip(
                    goalkeepers.tracker_id,
                    goalkeepers_team_id,
                    transformed_goalkeepers_positions,
                )
            ],
            "referees": [
                {"position": list(pos)} for pos in transformed_referees_positions
            ],
            "balls": [{"position": list(pos)} for pos in transformed_ball_positions],
        }

        self.all_frames.append(frame_data)

    def process_video(
        self, source_video_path: str, json_file_path: Optional[str] = None
    ) -> Iterator[np.ndarray]:
        """
        Process a video to generate radar view.

        Args:
            source_video_path: Path to the source video
            json_file_path: Path to save radar data as JSON

        Yields:
            Annotated frames with radar view
        """
        # Fit team classifier
        self.fit_team_classifier(source_video_path)

        # Reset counters and history
        self.total_frames = 0
        self.possession_counts = {TEAM_A_ID: 0, TEAM_B_ID: 0}
        self.all_frames = []

        # Process video frames
        logger.info("Starting main frame processing loop...")
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
        frame_index = 0

        for frame in frame_generator:
            # Detect pitch keypoints
            keypoints, transformer = self._detect_pitch_keypoints(frame)

            if transformer is None:
                logger.warning("No pitch keypoints detected, skipping frame.")
                continue


            # Detect goalkeepers and referees
            result = self.player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = self.player_tracker.update_with_detections(detections)

            # Extract players and classify teams
            players, players_team_id = self._detect_and_track_players(
                frame, detections
            )
            # Detect goalkeepers and referees
            goalkeepers, referees, goalkeepers_team_id = (
                self._detect_goalkeepers_and_referees(
                    detections, players, players_team_id
                )
            )

            # Transform player and goalkeeper positions
            transformed_players_positions = (
                transformer.transform_points(
                    players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                )
                / 100.0
            )

            transformed_goalkeepers_positions = (
                transformer.transform_points(
                    goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                )
                / 100.0
            )

            transformed_referees_positions = (
                transformer.transform_points(
                    referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                )
                / 100.0
            )

            # Calculate player speeds
            self._calculate_player_speeds(frame, players, transformed_players_positions)

            # Detect and track ball
            ball_detections = self._detect_and_track_ball(frame)
            transformed_ball_positions = (
                transformer.transform_points(
                    ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                )
                / 100.0
            )

            # Determine ball possession
            self._determine_ball_possession(
                frame,
                transformed_ball_positions,
                transformed_players_positions,
                players_team_id,
            )

            # Merge all detections for annotation
            all_detections = sv.Detections.merge(
                [players, goalkeepers, referees, ball_detections]
            )
            color_lookup = np.array(
                players_team_id.tolist()
                + goalkeepers_team_id.tolist()
                + [REFEREE_CLASS_ID] * len(referees)
                + [BALL_COLOR_ID] * len(ball_detections),
                dtype=int,
            )
            all_labels = [
                str(tracker_id) for tracker_id in all_detections.tracker_id
            ]

            # Annotate frame with player and ball detections
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, all_detections, custom_color_lookup=color_lookup
            )
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame,
                all_detections,
                all_labels,
                custom_color_lookup=color_lookup,
            )

            # Add radar overlay
            annotated_frame = create_radar_frame(
                annotated_frame, all_detections, color_lookup, all_labels, keypoints
            )

            # Save frame data for JSON export
            self._save_frame_data(
                frame_index,
                players,
                goalkeepers,
                players_team_id,
                goalkeepers_team_id,
                transformed_players_positions,
                transformed_goalkeepers_positions,
                transformed_referees_positions,
                transformed_ball_positions,
            )

            self.total_frames += 1
            frame_index += 1
            yield annotated_frame

        # Save radar data to JSON
        if json_file_path:
            logger.info(f"Saving radar data to {json_file_path}")
            converted_frames = convert_numpy_types(self.all_frames)
            with open(json_file_path, "w") as f:
                json.dump({"frames": converted_frames}, f, indent=2)


def run_radar(
    source_video_path: str, device: str, json_file_path: str
) -> Iterator[np.ndarray]:
    """
    Run radar view generation on a video and save data to JSON.

    Args:
        source_video_path: Path to the source video
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        json_file_path: Path to save radar data as JSON

    Yields:
        Annotated frames with radar view
    """
    radar = RadarView(
        player_model_path=PLAYER_DETECTION_MODEL_PATH,
        pitch_model_path=PITCH_DETECTION_MODEL_PATH,
        ball_model_path=BALL_DETECTION_MODEL_PATH,
        device=device,
    )
    yield from radar.process_video(source_video_path, json_file_path)
