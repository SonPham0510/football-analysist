
import argparse
import logging
from enum import Enum
from typing import  Optional





# Local imports
from utils.utils import save_video
from src.pitch_detection import run_pitch_detection
from src.player_detection import run_player_detection
from src.ball_detection import run_ball_detection
from src.player_tracking import run_player_tracking
from src.team_classfication import run_team_classification
from src.jersey_number import run_jersey_detection
from src.radar import run_radar

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Mode(Enum):
    """
    Enum class representing different modes of operation for football video analysis.
    
    Each mode corresponds to a specific type of analysis that can be performed
    on football videos, such as player detection, team classification, etc.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    JERSEY_DETECTION = 'JERSEY_DETECTION'
    RADAR = 'RADAR'


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    json_file_path: Optional[str] = None
) -> None:
    """
    Main function for running football video analysis.
    
    Args:
        source_video_path: Path to the input video
        target_video_path: Path where the annotated video will be saved
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        mode: Analysis mode to run
        json_file_path: Path to save additional data (required for RADAR mode)
        
    Raises:
        NotImplementedError: If the specified mode is not implemented
        ValueError: If required parameters are missing for a specific mode
    """
    logger.info(f"Running {mode.value} on {source_video_path} using {device}")
    
    # Validate inputs
    if mode == Mode.RADAR and not json_file_path:
        raise ValueError("JSON file path is required for RADAR mode")
    
    # Select processing mode and get frame generator
    frame_generator = None
    
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.JERSEY_DETECTION:
        frame_generator = run_jersey_detection(
            source_video_path=source_video_path,
            device=device
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            json_file_path=json_file_path
        )
    
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")
    
    # Save the annotated frames to the target video
    save_video(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        frame_generator=frame_generator,
        mode_name=mode.value
    )
    
    logger.info(f"Processing complete. Output saved to {target_video_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Football video analysis using deep learning'
    )
    parser.add_argument(
        '--source_video_path',
        type=str,
        required=True,
        help='Path to the input video file'
    )
    parser.add_argument(
        '--target_video_path',
        type=str,
        required=True,
        help='Path to save the output video file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run inference on (cpu, cuda, mps)'
    )
    parser.add_argument(
        '--mode',
        type=Mode,
        default=Mode.PLAYER_DETECTION,
        choices=list(Mode),
        help='Analysis mode to run'
    )
    parser.add_argument(
        '--json_file_path',
        type=str,
        default='output.json',
        help='Path to save additional data (required for RADAR mode)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        json_file_path=args.json_file_path
    )
