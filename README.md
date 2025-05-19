# Theis Soccer AI ⚽

## 💻 install
- Install package [uv](https://github.com/astral-sh/uv):

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Activate virtual environment:

```bash
# Create virtual environment.
uv venv --python 3.11

source .venv/bin/activate
```

- Install dependencies:

```bash
uv sync
```


```bash
git clone <url>
cd thesis
uv sync
./setup.sh



## 🛠️ modes

- `PITCH_DETECTION` - Detects the soccer field boundaries and key points in the video. 
Useful for identifying and visualizing the layout of the soccer pitch.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-pitch-detection.mp4 \
  --device mps --mode PITCH_DETECTION
  ```



- `PLAYER_DETECTION` - Detects players, goalkeepers, referees, and the ball in the 
video. Essential for identifying and tracking the presence of players and other 
entities on the field.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-detection.mp4 \
  --device mps --mode PLAYER_DETECTION
  ```

  https://github.com/user-attachments/assets/c36ea2c1-b03e-4ffe-81bd-27391260b187

- `BALL_DETECTION` - Detects the ball in the video frames and tracks its position. 
Useful for following ball movements throughout the match.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-ball-detection.mp4 \
  --device mps --mode BALL_DETECTION
  ```

  https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031

- `PLAYER_TRACKING` - Tracks players across video frames, maintaining consistent 
identification. Useful for following player movements and positions throughout the 
match.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-tracking.mp4 \
  --device mps --mode PLAYER_TRACKING
  ```
  
  https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839

- `TEAM_CLASSIFICATION` - Classifies detected players into their respective teams based 
on their visual features. Helps differentiate between players of different teams for 
analysis and visualization.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-team-classification.mp4 \
  --device mps --mode TEAM_CLASSIFICATION
  ```

  https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7

- `RADAR` - Combines pitch detection, player detection, tracking, and team 
classification to generate a radar-like visualization of player positions on the 
soccer field. Provides a comprehensive overview of player movements and team formations 
on the field.

  ```bash
  uv run  main.py --source_video_path data/0bfacc_10.mp4 \
  --target_video_path data/0bfacc_10radar.mp4 \
  --device cpu --mode RADAR --json_file_path data/0bfacc_10.json
  ```

  https://github.com/user-attachments/assets/263b4cd0-2185-4ed3-9be2-cf4d8f5bfa67

 - `POSSESSION_TRACKING` - Tracks players and goalkeepers across video frames,
 maintaining consistent identification. Helps identify and track players and goalkeepers
 on the field.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-possession-tracking.mp4 \
  --device mps --mode POSSESSION_TRACKING
  ```

  https://github.com/user-attachments/assets/d0d7f0a0-f7b7-4f1f-b6b3-c7d1b1a1e3c7

-"JERSEY_DETECTION" - Detects the jersey number of players in the video. Helps identify
players on the field and their respective jersey numbers.

  ```bash
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-jersey-detection.mp4 \
  --device mps --mode JERSEY_DETECTION
  ```

  https://github.com/user-attachments/assets/f0e7f9c0-f7b7-4f1f-b6b3-c7d1b1a1e3c7   



## 📊 Results
