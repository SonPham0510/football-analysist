import React from 'react';

const Project = () => {
  return (
    <div className="page-content">
      <h2>About the Project</h2>
      <p>This full-stack application demonstrates a powerful Football Analysis engine built in Python.</p>
      <h3>Core Features:</h3>
      <ul>
        <li><strong>Player & Ball Detection:</strong> Utilizes YOLOv8 models to accurately detect all entities on the field.</li>
        <li><strong>Player Tracking:</strong> Employs `ByteTrack` to maintain consistent IDs for players across video frames.</li>
        <li><strong>Team Classification:</strong> Uses a `SiglipVisionModel` and K-Means clustering to automatically assign players to their respective teams based on jersey color.</li>
        <li><strong>Pitch Detection:</strong> Identifies the pitch lines and keypoints to create a geometric mapping of the field.</li>
        <li><strong>Tactical Radar View:</strong> Transforms player coordinates onto a 2D top-down view of the pitch for tactical analysis.</li>
      </ul>
      <h3>Technology Stack:</h3>
      <ul>
        <li><strong>Backend:</strong> Python, FastAPI, Ultralytics YOLOv8, Supervision, OpenCV.</li>
        <li><strong>Frontend:</strong> React.js, Vite, Axios.</li>
      </ul>
    </div>
  );
};

export default Project;