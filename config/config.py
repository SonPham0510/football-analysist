
import os
import supervision as sv

from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from pitch_annotator.soccer import SoccerPitchConfiguration



####MODEL PATHS####
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'football-player-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'football-ball-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'football-pitch-detection.pt')



###CLASS IDS###
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
BALL_COLOR_ID = 4

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#FFFFFF']

#### ANNOTATORS ####
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)





