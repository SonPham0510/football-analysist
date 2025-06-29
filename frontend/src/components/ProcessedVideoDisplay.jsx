import React from 'react';
import VideoPlayer from './VideoPlayer';
import './ProcessedVideoDisplay.css';

const ProcessedVideoDisplay = ({ videoInfo, processedVideo }) => {
    if (!videoInfo || !processedVideo) return null;

    const isCloudReady = videoInfo.status?.ready_to_view && videoInfo.cloudinary;
    const isLocalFallback = videoInfo.status?.processing_complete && !videoInfo.status?.cloud_upload_complete;

    // If cloud is ready, show cloud video
    if (isCloudReady) {
        return (
            <div className="processed-video-section success">
                <div className="section-header">
                    <h3>🎉 Analysis Complete!</h3>
                    <div className="success-badge">
                        <span>✅ Processed</span>
                        <span>☁️ Cloud Ready</span>
                    </div>
                </div>

                <div className="video-display">
                    <VideoPlayer
                        src={videoInfo.cloudinary.direct_url}
                        title={`Processed: ${videoInfo.video_name}`}
                        className="main-player"
                    />
                </div>
            </div>
        );
    }

    // If only local processing is complete, show local video
    if (isLocalFallback) {
        return (
            <div className="processed-video-section fallback">
                <div className="section-header">
                    <h3>⚠️ Processing Complete </h3>
                    <div className="warning-badge">
                        <span>✅ Processed</span>
                        <span>💻 Local Only</span>
                    </div>
                </div>

                <div className="video-display">
                    <VideoPlayer
                        src={`http://localhost:8000/video/${processedVideo}`}
                        title={`Processed: ${processedVideo}`}
                        className="main-player"
                    />
                </div>
            </div>
        );
    }

    // Fallback - just show the video if we have processed video
    return (
        <div className="processed-video-section">
            <div className="section-header">
                <h3>🎬 Processed Video</h3>
            </div>

            <div className="video-display">
                <VideoPlayer
                    src={videoInfo.cloudinary.direct_url}
                    title={`Processed: ${videoInfo.video_name}`}
                    className="main-player"
                />

            </div>
        </div>
    );
};

export default ProcessedVideoDisplay;
