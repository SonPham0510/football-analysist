# Theis Soccer AI ⚽

## 💻 install
- Install package [uv](https://github.com/astral-sh/uv):

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Clone the repository and set up the environment:

```bash
# Clone the repository
git clone <url>
cd thesis
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

# Add executable permission to setup script (if exists)
```bash
chmod +x ./setup.sh
```
# Run setup script (if exists)
```bash
./setup.sh
```
# Run function of project

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
  --device cuda --mode BALL_DETECTION
```

  https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031

- `PITCH_DETECTION` - Detects the soccer field boundaries and key points in the video. 
Useful for identifying and visualizing the layout of the soccer pitch.

```bash  
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-pitch-detection.mp4 \
  --device cuda --mode PITCH_DETECTION
```

- `PLAYER_TRACKING` - Tracks players across video frames, maintaining consistent 
identification. Useful for following player movements and positions throughout the 
match.

```bash  
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-player-tracking.mp4 \
  --device cuda --mode PLAYER_TRACKING
```
  
  https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839

- `TEAM_CLASSIFICATION` - Classifies detected players into their respective teams based 
on their visual features. Helps differentiate between players of different teams for 
analysis and visualization.

```bash  
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-team-classification.mp4 \
  --device cuda --mode TEAM_CLASSIFICATION
```

  https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7

- `RADAR_CHART` - Generates a radar chart with player statistics, such as speed,
distance covered, and other performance metrics. Useful for visualizing player
performance in a comprehensive manner.

```bash  
  uv run main.py --source_video_path data/2e57b9_0.mp4 \
  --target_video_path data/2e57b9_0-radar-chart.mp4 \
  --device cuda --mode RADAR_CHART --json_path data/2e57b9_0._radar.json
  
```

<video width="600" controls>
  <source src="https://github.com/user-attachments/assets/2e57b9_0-radar_3.mp4" type="video/mp4">
</video>


## 🚀 Web Interface

A React web application is provided to upload a video and view the processed
result. The frontend lives in the `frontend/` folder and uses Vite for
development.

### Start the backend

Run the FastAPI server from the repository root:

```bash
uvicorn backend.main:app --reload
```

### Start the frontend

Install the Node dependencies once and then launch the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`. Use the form on the
"Solution" page to upload a video and choose one or more analysis modes. Each
mode will produce a separate processed video when the backend finishes
processing.

