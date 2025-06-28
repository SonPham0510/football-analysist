import React from 'react';
import VideoPlayer from './VideoPlayer';
import './VideoPreview.css';

const VideoPreview = ({ previewUrl, selectedFile }) => {
    if (!previewUrl || !selectedFile) return null;

    return (
        <div className="preview-section">
            <div className="preview-header">
                <h3>📹 Selected Video Preview</h3>
                <div className="file-info">
                    <span className="file-name">{selectedFile.name}</span>
                    <span className="file-size">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                </div>
            </div>

            <div className="preview-content">
                <VideoPlayer
                    src={previewUrl}
                    title={selectedFile.name}
                    className="preview-player"
                />
            </div>
        </div>
    );
};

export default VideoPreview;
