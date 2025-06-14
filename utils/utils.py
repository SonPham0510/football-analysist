import json
import logging
import os
from typing import Any, Iterator, List, Optional

import numpy as np
import supervision as sv
from tqdm import tqdm

from config.config import COLORS, CONFIG,BALL_COLOR_ID
from pitch_annotator.soccer import draw_pitch, draw_points_with_labels_on_pitch
from ViewTransform.view_tranform import ViewTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global storage for frames to be exported to JSON
all_frames = []


def validate_video_path(path: str) -> bool:
    """
    Validate that a video file exists at the specified path.

    Args:
        path: Path to the video file to validate

    Returns:
        True if the file exists, False otherwise
    """
    if not os.path.exists(path):
        logger.error(f"Video file not found: {path}")
        return False
    return True


def validate_model_path(path: str) -> bool:
    """
    Validate that a model file exists at the specified path.

    Args:
        path: Path to the model file to validate

    Returns:
        True if the file exists, False otherwise
    """
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return False
    return True


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        pt1: First point coordinates (x, y)
        pt2: Second point coordinates (x, y)

    Returns:
        Euclidean distance between the two points
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from a frame based on detected bounding boxes.

    Args:
        frame: The frame from which to extract crops
        detections: Detected objects with bounding boxes

    Returns:
        List of cropped images
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def convert_numpy_types(data: Any) -> Any:
    """
    Recursively convert numpy types to native Python types.

    This function handles numpy arrays, scalars, and nested structures
    such as lists and dictionaries containing numpy types.

    Args:
        data: Input data that may contain numpy types

    Returns:
        Data with numpy types converted to Python native types
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, np.generic):  # Handle numpy scalar types
        return data.item()
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]

    return data  # Return native types unchanged


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Render a top-down radar view of the football pitch with player positions.

    Args:
        detections: Object detections (players, goalkeepers, etc.)
        keypoints: Pitch keypoints for coordinate transformation
        color_lookup: Color indices for the detected objects
        labels: Optional labels for the detected objects

    Returns:
        Rendered radar image
    """
    # Filter valid keypoints
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

    # Create view transformer to convert frame coordinates to pitch coordinates
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )

    # Transform detection coordinates to pitch space
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # Draw base pitch
    radar = draw_pitch(config=CONFIG)

    # Draw entities by team/color
    for color_id in range(5):  # Handle up to 5 different colors
        # Filter detections by color
        team_mask = color_lookup == color_id
        team_xy = transformed_xy[team_mask]

        # Filter labels by team if labels are provided
        team_labels = None
        if labels:
            team_labels = [labels[i] for i in range(len(labels)) if team_mask[i]]

        # Draw points for this team/color
        radar = draw_points_with_labels_on_pitch(
            config=CONFIG,
            xy=team_xy,
            face_color=sv.Color.from_hex(COLORS[color_id]),
            radius=5 if color_id == BALL_COLOR_ID else 20,  # Smaller radius for ball
            pitch=radar,
            labels=team_labels,
        )

    return radar


def create_radar_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    color_lookup: np.ndarray,
    labels: List[str],
    keypoints: sv.KeyPoints,
) -> np.ndarray:
    """
    Create a frame with radar overlay.

    This function:
    1. Renders the radar view
    2. Resizes it to fit as an overlay
    3. Overlays it onto the current frame

    Args:
        frame: Original video frame
        detections: Object detections
        color_lookup: Color indices for the detections
        labels: Labels for the detections
        keypoints: Pitch keypoints for coordinate transformation

    Returns:
        Frame with radar overlay
    """
    # Get frame dimensions
    h, w, _ = frame.shape

    # Render and resize radar
    radar = render_radar(detections, keypoints, color_lookup, labels)
    radar = sv.resize_image(radar, (w // 2, h // 2))

    # Position radar overlay
    radar_h, radar_w, _ = radar.shape
    rect = sv.Rect(
        x=w // 2 - radar_w // 2,  # Center horizontally
        y=h - radar_h,  # Bottom of the frame
        width=radar_w,
        height=radar_h,
    )

    # Overlay radar with semi-transparency
    annotated_frame = sv.draw_image(frame, radar, opacity=0.5, rect=rect)

    return annotated_frame


def save_all_frames_to_json(json_file_path: str) -> None:
    """
    Save all collected frames to a JSON file.

    Args:
        json_file_path: Path to the output JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    converted_frames = convert_numpy_types(all_frames)

    # Write to JSON file with indentation for readability
    with open(json_file_path, "w") as json_file:
        json.dump({"frames": converted_frames}, json_file, indent=4)

    logger.info(f"Frame data saved to {json_file_path}")


def save_video(
    source_video_path: str,
    target_video_path: str,
    frame_generator: Iterator[np.ndarray],
    mode_name: str,
) -> None:
    """
    Save processed frames to a video file.

    Args:
        source_video_path: Path to the source video (for metadata)
        target_video_path: Path where the output video will be saved
        frame_generator: Iterator yielding processed video frames
        mode_name: Name of the processing mode (for progress display)
    """
    try:
        # Get video metadata from source
        video_info = sv.VideoInfo.from_video_path(source_video_path)

        # Write frames to output video
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frame_generator, desc=f"Processing {mode_name}"):
                sink.write_frame(frame)

        logger.info(f"Video saved to {target_video_path}")
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
