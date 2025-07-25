import json
import logging
import os
from typing import Any, Iterator, List, Optional, Dict
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

from config.config import COLORS, CONFIG, BALL_COLOR_ID
from cloudinary_config import CloudinaryManager
from pitch_annotator.soccer import draw_pitch, draw_points_with_labels_on_pitch
from ViewTransform.view_tranform import ViewTransformer
from cloudinary_config import CloudinaryManager

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
    for color_id in range(5):
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
            radius=5 if color_id == BALL_COLOR_ID else 30,  # Smaller radius for ball
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
                logger.info(f"Writing frame of shape {frame.shape} to video")
                sink.write_frame(frame)

        logger.info(f"Video saved to {target_video_path}")
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")


class VideoUtils:
    @staticmethod
    def upload_processed_video_to_cloud(
        local_path: str, video_name: str
    ) -> Dict[str, Any]:
        """
        Upload processed video to Cloudinary and return cloud info
        """
        try:
            if not os.path.exists(local_path):
                return {"success": False, "error": "Local video file not found"}

            # Generate public_id from video name (remove extension)
            public_id = Path(video_name).stem

            # Upload to Cloudinary
            result = CloudinaryManager.upload_video(
                file_path=local_path, public_id=public_id, folder="football_analysis"
            )

            if result["success"]:
                logger.info(f"Successfully uploaded {video_name} to Cloudinary")
                return {
                    "success": True,
                    "cloudinary_url": result["secure_url"],
                    "public_id": result["public_id"],
                    "player_url": CloudinaryManager.get_player_embed_url(
                        result["public_id"]
                    ),
                    "direct_url": CloudinaryManager.get_video_url(result["public_id"]),
                }
            else:
                logger.error(f"Failed to upload {video_name}: {result['error']}")
                return result

        except Exception as e:
            logger.error(f"Error uploading video {video_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_cloud_video_info(public_id: str) -> Dict[str, Any]:
        """
        Get video information from Cloudinary
        """
        try:
            direct_url = CloudinaryManager.get_video_url(public_id)
            player_url = CloudinaryManager.get_player_embed_url(public_id)

            return {
                "success": True,
                "public_id": public_id,
                "direct_url": direct_url,
                "player_url": player_url,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def cleanup_local_file(file_path: str) -> bool:
        """
        Remove local file after uploading to cloud
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up local file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")
            return False

    @staticmethod
    def upload_csv_to_cloud(local_csv_path: str, csv_name: str) -> Dict[str, Any]:
        """
        Upload CSV file to Cloudinary and return cloud info
        """
        try:
            if not os.path.exists(local_csv_path):
                return {"success": False, "error": "Local CSV file not found"}

            # Generate public_id from CSV name (remove extension)
            public_id = Path(csv_name).stem

            # Upload to Cloudinary
            result = CloudinaryManager.upload_csv(
                file_path=local_csv_path,
                public_id=public_id,
                folder="football_analysis/csv",
            )

            if result["success"]:
                logger.info(f"Successfully uploaded {csv_name} to Cloudinary")
                return {
                    "success": True,
                    "cloudinary_url": result["secure_url"],
                    "public_id": result["public_id"],
                    "direct_url": result["url"],
                    "bytes": result.get("bytes", 0),
                }
            else:
                logger.error(f"Failed to upload {csv_name}: {result['error']}")
                return result

        except Exception as e:
            logger.error(f"Error uploading CSV {csv_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def upload_complete_match_data(
        video_path: str, csv_path: str, match_name: str
    ) -> Dict[str, Any]:
        """
        Upload both video and CSV for a complete match analysis
        """
        try:
            results = {}

            # Upload video first
            video_result = VideoUtils.upload_processed_video_to_cloud(
                video_path, f"{match_name}.mp4"
            )
            results["video"] = video_result

            # Upload CSV if exists
            if os.path.exists(csv_path):
                csv_result = VideoUtils.upload_csv_to_cloud(
                    csv_path, f"{match_name}.csv"
                )
                results["csv"] = csv_result
            else:
                results["csv"] = {"success": False, "error": "CSV file not found"}

            # Overall success depends on both uploads
            overall_success = video_result.get("success", False) and results["csv"].get(
                "success", False
            )

            return {
                "success": overall_success,
                "match_name": match_name,
                "video": video_result,
                "csv": results["csv"],
                "message": "Complete match data uploaded successfully"
                if overall_success
                else "Partial upload completed",
            }

        except Exception as e:
            logger.error(f"Error uploading complete match data: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def download_csv_from_cloud(public_id: str, local_path: str) -> bool:
        """
        Download CSV file from Cloudinary for local analysis
        """
        try:
            return CloudinaryManager.download_csv(public_id, local_path)
        except Exception as e:
            logger.error(f"Error downloading CSV {public_id}: {str(e)}")
            return False
