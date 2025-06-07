from typing import List

import numpy as np


def estimate_speed(position_history: List, frame_rate: float) -> float:
    """
    Estimate the speed of a player given their position history.
    :param position_history: List of (x, y) positions.
    :param frame_rate: Frame rate of the video (frames per second).
    :return: Speed in meters per second (m/s).
    """
    if len(position_history) < 2:
        return 0.0  # Not enough data to calculate speed

    # Calculate distance between the last two positions
    delta_x = position_history[-1][0] - position_history[-2][0]
    delta_y = position_history[-1][1] - position_history[-2][1]
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # Time between two frames
    time = 1 / frame_rate

    # Speed = distance / time
    speed = distance / time
    return speed
