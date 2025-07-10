# import json
import logging
import csv
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

# Configure logger
logger = logging.getLogger(__name__)

# Constants for team identification
TEAM_A_ID = 0
TEAM_B_ID = 1


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
        try:
            logger.info("Initializing RadarView...")

            self.player_model_path = player_model_path
            self.pitch_model_path = pitch_model_path
            self.ball_model_path = ball_model_path
            self.device = device
            self.min_consecutive_frames = min_consecutive_frames
            self.ball_buffer_size = ball_buffer_size
            self.frames_to_fit = frames_to_fit

            # Tracking and statistics
            self.position_history = defaultdict(list)
            self.possession_counts = {TEAM_A_ID: 0, TEAM_B_ID: 0}
            self.total_frames = 0
            self.all_frames = []  # For JSON export

            # CSV data storage
            self.csv_data = []
            self.frame_rate = 30.0  # Will be updated from video
            self.current_phase_id = 0
            self.last_ball_possession_player = None

            # Ball tracking
            self.last_ball_detections = None
            self.ball_missing_frames = 0
            self.ball_missing_threshold = 15

            # Initialize models and components
            self._load_models()
            self._initialize_components()

            logger.info("RadarView initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RadarView: {e}")
            raise

    def _load_models(self) -> None:
        """Load detection models."""
        try:
            logger.info("Loading detection models...")

            logger.debug(f"Loading player model from: {self.player_model_path}")
            self.player_model = YOLO(self.player_model_path).to(device=self.device)

            logger.debug(f"Loading pitch model from: {self.pitch_model_path}")
            self.pitch_model = YOLO(self.pitch_model_path).to(device=self.device)

            logger.debug(f"Loading ball model from: {self.ball_model_path}")
            self.ball_model = YOLO(self.ball_model_path).to(device=self.device)

            logger.info(f"All models loaded successfully on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize trackers and other components."""
        try:
            logger.info("Initializing tracking components...")

            self.player_tracker = sv.ByteTrack(
                minimum_consecutive_frames=self.min_consecutive_frames
            )
            logger.info(
                f"Player tracker initialized with min_frames: {self.min_consecutive_frames}"
            )

            self.ball_tracker = BallTracker(buffer_size=self.ball_buffer_size)
            logger.info(
                f"Ball tracker initialized with buffer_size: {self.ball_buffer_size}"
            )

            self.team_classifier = TeamClassifier(device=self.device)
            logger.info("Team classifier initialized")

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def fit_team_classifier(self, source_video_path: str) -> None:
        """
        Fit the team classifier using player crops from the video.

        Args:
            source_video_path: Path to the source video
        """
        try:
            logger.info(
                f"Starting team classifier fitting for video: {source_video_path}"
            )

            frame_generator = sv.get_video_frames_generator(
                source_path=source_video_path, stride=STRIDE
            )

            crops = []
            frame_count = 0

            for frame in tqdm(frame_generator, desc="Collecting crops"):
                try:
                    frame_count += 1

                    result = self.player_model(frame, imgsz=1280, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(result)

                    frame_crops = get_crops(
                        frame, detections[detections.class_id == PLAYER_CLASS_ID]
                    )
                    crops.extend(frame_crops)

                    logger.debug(
                        f"Frame {frame_count}: Collected {len(frame_crops)} crops"
                    )

                    # Limit number of crops to avoid excessive memory usage
                    if len(crops) > 2000:
                        logger.info("Reached crop limit of 2000, stopping collection")
                        break

                except Exception as e:
                    logger.warning(
                        f"Error processing frame {frame_count} for team fitting: {e}"
                    )
                    continue

            logger.info(f"Collected {len(crops)} crops from {frame_count} frames")

            if len(crops) == 0:
                raise ValueError("No player crops collected for team classification")

            self.team_classifier.fit(crops)
            logger.info("Team classifier fitted successfully")

        except Exception as e:
            logger.error(f"Failed to fit team classifier: {e}")
            raise

    def _detect_pitch_keypoints(
        self, frame: np.ndarray
    ) -> Tuple[Optional[sv.KeyPoints], Optional[ViewTransformer]]:
        """
        Detect pitch keypoints and create a view transformer.

        Args:
            frame: Current video frame

        Returns:
            Tuple containing keypoints and view transformer
        """
        try:
            result = self.pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)

            # Create a ViewTransformer using the keypoints
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

            if mask.sum() < 4:
                logger.warning(f"Not enough keypoints detected: {mask.sum()}/4 minimum")
                return None, None

            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32),
            )

            logger.info(f"Pitch keypoints detected: {mask.sum()} valid points")
            return keypoints, transformer

        except Exception as e:
            logger.error(f"Error detecting pitch keypoints: {e}")
            return None, None

    def _detect_and_track_players(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> Tuple[sv.Detections, np.ndarray]:
        """
        Detect and track players, assigning team IDs

        Args:
            frame: Current video frame

        Returns:
            Tuple containing:
                - player detections
                - team IDs

        """
        try:
            # Filter out players
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            logger.info(f"Detected {len(players)} players")

            if len(players) == 0:
                logger.info("No players detected in frame")
                return players, np.array([])

            # Classify players by team
            crops = get_crops(frame, players)
            players_team_id = self.team_classifier.predict(crops)

            logger.debug(f"Team classification: {len(players_team_id)} assignments")
            return players, players_team_id

        except Exception as e:
            logger.error(f"Error detecting and tracking players: {e}")
            return sv.Detections.empty(), np.array([])

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
        try:
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            logger.info(
                f"Detected {len(goalkeepers)} goalkeepers, {len(referees)} referees"
            )

            # Resolve goalkeeper team IDs
            goalkeepers_team_id = np.array([], dtype=int)
            if len(goalkeepers) > 0 and len(players) > 0:
                try:
                    goalkeepers_team_id = resolve_goalkeepers_team_id(
                        players, players_team_id, goalkeepers
                    )
                    logger.info(
                        f"Resolved {len(goalkeepers_team_id)} goalkeeper team IDs"
                    )
                except Exception as e:
                    logger.warning(f"Failed to resolve goalkeeper team IDs: {e}")

            return goalkeepers, referees, goalkeepers_team_id

        except Exception as e:
            logger.error(f"Error detecting goalkeepers and referees: {e}")
            return sv.Detections.empty(), sv.Detections.empty(), np.array([])

    def _detect_and_track_ball(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect and track the ball.

        Args:
            frame: Current video frame

        Returns:
            Ball detections
        """
        try:
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
                    logger.info(
                        f"Ball missing for {self.ball_missing_frames} frames, using last detection"
                    )
                else:
                    self.last_ball_detections = None
                    self.ball_missing_frames = 0
                    logger.debug("Ball lost, resetting tracking")
            else:
                self.last_ball_detections = ball_detections
                self.ball_missing_frames = 0
                logger.debug("Ball detected and tracked")

            # Ensure ball has tracker_id
            if ball_detections.tracker_id is None:
                ball_detections.tracker_id = np.arange(len(ball_detections))

            return ball_detections

        except Exception as e:
            logger.error(f"Error detecting and tracking ball: {e}")
            return sv.Detections.empty()

    def _calculate_player_speeds(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        transformed_positions: np.ndarray,
        frame_rate: float = 30.0,
    ) -> dict:
        """
        Calculate and annotate player speeds.

        Args:
            frame: Current video frame for annotation
            players: Player detections
            transformed_positions: Transformed player positions
            frame_rate: Video frame rate

        Returns:
            Dictionary mapping player_id to speed
        """
        player_speeds = {}

        try:
            for tracker_id, position, bbox in zip(
                players.tracker_id, transformed_positions, players.xyxy
            ):
                try:
                    # Update position history
                    self.position_history[tracker_id].append(position)
                    if (
                        len(self.position_history[tracker_id]) > 10
                    ):  # Keep last 10 positions
                        self.position_history[tracker_id].pop(0)

                    # Calculate speed
                    speed = estimate_speed(
                        self.position_history[tracker_id], frame_rate
                    )
                    player_speeds[tracker_id] = speed

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

                except Exception as e:
                    logger.warning(
                        f"Error calculating speed for player {tracker_id}: {e}"
                    )
                    player_speeds[tracker_id] = 0.0

            logger.info(f"Calculated speeds for {len(player_speeds)} players")
            return player_speeds

        except Exception as e:
            logger.error(f"Error in player speed calculation: {e}")
            return {}

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

        try:
            if (
                len(transformed_ball_positions) > 0
                and len(transformed_players_positions) > 0
            ):
                ball_pos = transformed_ball_positions[0]
                min_dist = float("inf")
                closest_player_team_id: Optional[int] = None

                # Find the closest player in pitch space
                for pos, team_id in zip(transformed_players_positions, players_team_id):
                    try:
                        dist = euclidean_distance(ball_pos, pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_player_team_id = int(team_id)
                    except Exception as e:
                        logger.warning(
                            f"Error calculating distance for team {team_id}: {e}"
                        )
                        continue

                # Update possession counter
                if (
                    closest_player_team_id is not None
                    and closest_player_team_id in self.possession_counts
                ):
                    self.possession_counts[closest_player_team_id] += 1
                    current_ball_possession_team = closest_player_team_id
                    logger.info(
                        f"Ball possession: Team {current_ball_possession_team} (distance: {min_dist:.2f}m)"
                    )

            # Annotate possession info
            self._annotate_possession_info(frame, current_ball_possession_team)

        except Exception as e:
            logger.error(f"Error determining ball possession: {e}")

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
        try:
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

        except Exception as e:
            logger.error(f"Error annotating possession info: {e}")

    def _save_csv_data(
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
        player_speeds: dict,
    ) -> None:
        """
        Save frame data for CSV export with format: time,teamId,playerId,x,y,hasBall,phaseId,speed

        Args:
            frame_index: Current frame index
            players: Player detections
            goalkeepers: Goalkeeper detections
            players_team_id: Team IDs for players
            goalkeepers_team_id: Team IDs for goalkeepers
            transformed_players_positions: Transformed player positions
            transformed_goalkeepers_positions: Transformed goalkeeper positions
            transformed_ball_positions: Transformed ball positions
            player_speeds: Dictionary mapping player_id to speed
        """
        try:
            time_stamp = frame_index / self.frame_rate

            # Determine ball possession for each player
            ball_pos = None
            if len(transformed_ball_positions) > 0:
                ball_pos = transformed_ball_positions[0]

            # Save player tracking data
            for tracker_id, team_id, pos in zip(
                players.tracker_id, players_team_id, transformed_players_positions
            ):
                try:
                    has_ball = False
                    if ball_pos is not None:
                        # Check if this player is closest to the ball (within 2 meters)
                        dist = euclidean_distance(ball_pos, pos)
                        has_ball = dist < 2.0

                    speed = player_speeds.get(tracker_id, 0.0)

                    self.csv_data.append(
                        {
                            "time": time_stamp,
                            "teamId": int(team_id),
                            "playerId": int(tracker_id),
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "hasBall": has_ball,
                            "phaseId": self.current_phase_id,
                            "speed": speed,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error saving data for player {tracker_id}: {e}")
                    continue

            logger.debug(
                f"Saved CSV data for frame {frame_index} with {len(players)} players"
            )

        except Exception as e:
            logger.error(f"Error saving CSV data for frame {frame_index}: {e}")

    def _write_csv_file(self, csv_file_path: str) -> None:
        """Write CSV file with tracking data."""
        try:
            if not self.csv_data:
                logger.warning("No CSV data to write")
                return

            with open(csv_file_path, "w", newline="") as f:
                fieldnames = [
                    "time",
                    "teamId",
                    "playerId",
                    "x",
                    "y",
                    "hasBall",
                    "phaseId",
                    "speed",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)

            logger.info(f"CSV tracking data saved to {csv_file_path}")
            logger.info(f"Total records: {len(self.csv_data)}")
            logger.info(f"Total phases detected: {self.current_phase_id + 1}")

        except Exception as e:
            logger.error(f"Failed to write CSV file {csv_file_path}: {e}")
            raise

    def _update_phase_tracking(
        self,
        current_ball_possession_player: Optional[int],
        transformed_ball_positions: np.ndarray,
        transformed_players_positions: np.ndarray,
        players_team_id: np.ndarray,
        players_tracker_id: np.ndarray,
    ) -> None:
        """
        Update phase tracking when ball possession changes.

        Args:
            current_ball_possession_player: Current player with ball possession
            transformed_ball_positions: Transformed ball positions
            transformed_players_positions: Transformed player positions
            players_team_id: Team IDs for players
            players_tracker_id: Player tracker IDs
        """
        try:
            if (
                len(transformed_ball_positions) > 0
                and len(transformed_players_positions) > 0
            ):
                ball_pos = transformed_ball_positions[0]
                min_dist = float("inf")
                closest_player_id = None

                # Find the closest player to the ball
                for pos, tracker_id in zip(
                    transformed_players_positions, players_tracker_id
                ):
                    try:
                        dist = euclidean_distance(ball_pos, pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_player_id = tracker_id
                    except Exception as e:
                        logger.warning(
                            f"Error calculating distance for player {tracker_id}: {e}"
                        )
                        continue

                # Check for phase change
                if (
                    closest_player_id is not None
                    and self.last_ball_possession_player is not None
                    and self.last_ball_possession_player != closest_player_id
                ):
                    self.current_phase_id += 1
                    logger.info(
                        f"Phase change detected: Player {closest_player_id} now has ball (Phase {self.current_phase_id})"
                    )

                self.last_ball_possession_player = closest_player_id

        except Exception as e:
            logger.error(f"Error updating phase tracking: {e}")

    def process_video(
        self, source_video_path: str, csv_file_path: Optional[str] = None
    ) -> Iterator[np.ndarray]:
        """
        Process a video to generate radar view.

        Args:
            source_video_path: Path to the source video
            csv_file_path: Path to save radar data as CSV


        Yields:
            Annotated frames with radar view
        """
        try:
            logger.info(f"Starting video processing: {source_video_path}")

            # Get video properties
            video_info = sv.VideoInfo.from_video_path(source_video_path)
            self.frame_rate = video_info.fps
            logger.info(
                f"Video properties - FPS: {self.frame_rate}, Duration: {video_info.total_frames / self.frame_rate:.2f}s"
            )

            # Fit team classifier
            self.fit_team_classifier(source_video_path)

            # Reset counters and history
            self.possession_counts = {TEAM_A_ID: 0, TEAM_B_ID: 0}
            self.all_frames = []
            self.csv_data = []
            self.current_phase_id = 0
            self.last_ball_possession_player = None

            # Process video frames
            logger.info("Starting main frame processing loop...")
            frame_generator = sv.get_video_frames_generator(
                source_path=source_video_path
            )
            frame_index = 0
            successful_frames = 0
            skipped_frames = 0

            for frame in frame_generator:
                try:
                    # Detect pitch keypoints
                    keypoints, transformer = self._detect_pitch_keypoints(frame)

                    if transformer is None:
                        logger.debug(
                            f"Frame {frame_index}: No pitch keypoints detected, skipping"
                        )
                        skipped_frames += 1
                        frame_index += 1
                        continue

                    # Detect all objects
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

                    # Transform positions
                    transformed_players_positions = (
                        transformer.transform_points(
                            players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        )
                        / 100.0
                    )

                    transformed_goalkeepers_positions = (
                        transformer.transform_points(
                            goalkeepers.get_anchors_coordinates(
                                sv.Position.BOTTOM_CENTER
                            )
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
                    player_speeds = self._calculate_player_speeds(
                        frame, players, transformed_players_positions, self.frame_rate
                    )

                    # Detect and track ball
                    ball_detections = self._detect_and_track_ball(frame)
                    transformed_ball_positions = (
                        transformer.transform_points(
                            ball_detections.get_anchors_coordinates(
                                sv.Position.BOTTOM_CENTER
                            )
                        )
                        / 100.0
                    )

                    # Update phase tracking
                    self._update_phase_tracking(
                        None,
                        transformed_ball_positions,
                        transformed_players_positions,
                        players_team_id,
                        players.tracker_id,
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
                        str(tracker_id) if tracker_id is not None else "?"
                        for tracker_id in all_detections.tracker_id
                    ]

                    # Annotate frame
                    annotated_frame = frame.copy()
                    annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                        annotated_frame,
                        all_detections,
                        custom_color_lookup=color_lookup,
                    )
                    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                        annotated_frame,
                        all_detections,
                        all_labels,
                        custom_color_lookup=color_lookup,
                    )

                    # Add radar overlay
                    annotated_frame = create_radar_frame(
                        annotated_frame,
                        all_detections,
                        color_lookup,
                        all_labels,
                        keypoints,
                    )

                    # Save CSV data
                    self._save_csv_data(
                        frame_index,
                        players,
                        goalkeepers,
                        players_team_id,
                        goalkeepers_team_id,
                        transformed_players_positions,
                        transformed_goalkeepers_positions,
                        transformed_referees_positions,
                        transformed_ball_positions,
                        player_speeds,
                    )

                    successful_frames += 1
                    frame_index += 1

                    # Log progress every 100 frames
                    if frame_index % 100 == 0:
                        logger.info(
                            f"Processed {frame_index} frames ({successful_frames} successful, {skipped_frames} skipped)"
                        )

                    yield annotated_frame

                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}")
                    skipped_frames += 1
                    frame_index += 1
                    continue

            # Write CSV file
            if csv_file_path:
                self._write_csv_file(csv_file_path)

            logger.info(
                f"Video processing completed: {successful_frames} successful frames, {skipped_frames} skipped frames"
            )

        except Exception as e:
            logger.error(f"Critical error in video processing: {e}")
            raise


def run_radar(
    source_video_path: str, device: str, csv_file_path: Optional[str] = None
) -> Iterator[np.ndarray]:
    """
    Run radar view generation on a video and save data to CSV.

    Args:
        source_video_path: Path to the source video
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        json_file_path: Path to save radar data as JSON (optional)
        csv_file_path: Path to save CSV data (optional, defaults to source_video_path with .csv extension)

    Yields:
        Annotated frames with radar view
    """
    try:
        logger.info(f"Starting radar processing with device: {device}")

        radar = RadarView(
            player_model_path=PLAYER_DETECTION_MODEL_PATH,
            pitch_model_path=PITCH_DETECTION_MODEL_PATH,
            ball_model_path=BALL_DETECTION_MODEL_PATH,
            device=device,
        )

        yield from radar.process_video(source_video_path, csv_file_path)

        logger.info("Radar processing completed successfully")

    except Exception as e:
        logger.error(f"Failed to run radar processing: {e}")
        raise
