import React from 'react';
import VideoPlayer from './VideoPlayer';
import './ProcessedVideoDisplay.css';

const ProcessedVideoDisplay = ({ videoInfo, processedVideo }) => {
    if (!videoInfo || !processedVideo) return null;

    const isCloudReady = videoInfo.status?.ready_to_view && videoInfo.cloudinary;
    const isLocalFallback = videoInfo.status?.processing_complete && !videoInfo.status?.cloud_upload_complete;

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
    };

    if (isCloudReady) {
        return (
            <div className="processed-video-section success">
                <div className="section-header">
                    <h3>🎉 Analysis Complete & Cloud Ready!</h3>
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

                <div className="video-actions">
                    <div className="action-group">
                        <h4>🔗 Share & Download</h4>
                        <div className="link-item">
                            <span className="link-label">📱 Direct Video URL:</span>
                            <button
                                onClick={() => copyToClipboard(videoInfo.cloudinary.direct_url)}
                                className="copy-btn"
                            >
                                📋 Copy Link
                            </button>
                            <a
                                href={videoInfo.cloudinary.direct_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="external-link"
                            >
                                🔗 Open
                            </a>
                        </div>

                        {videoInfo.cloudinary.player_url && (
                            <div className="link-item">
                                <span className="link-label">🎬 Player URL:</span>
                                <button
                                    onClick={() => copyToClipboard(videoInfo.cloudinary.player_url)}
                                    className="copy-btn"
                                >
                                    📋 Copy Link
                                </button>
                                <a
                                    href={videoInfo.cloudinary.player_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="external-link"
                                >
                                    🔗 Open
                                </a>
                            </div>
                        )}
                    </div>
                </div>

                <div className="video-metadata">
                    <h4>📊 Processing Details</h4>
                    <div className="metadata-grid">
                        <div className="metadata-item">
                            <span className="metadata-label">Video Name:</span>
                            <span className="metadata-value">{videoInfo.video_name}</span>
                        </div>
                        <div className="metadata-item">
                            <span className="metadata-label">Analysis Mode:</span>
                            <span className="metadata-value">{videoInfo.processing_mode?.replace('_', ' ')}</span>
                        </div>
                        <div className="metadata-item">
                            <span className="metadata-label">File Size:</span>
                            <span className="metadata-value">
                                {videoInfo.file_size ? `${(videoInfo.file_size / 1024 / 1024).toFixed(2)} MB` : 'Unknown'}
                            </span>
                        </div>
                        <div className="metadata-item">
                            <span className="metadata-label">Status:</span>
                            <span className="metadata-value success">✅ Successfully Processed</span>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (isLocalFallback) {
        return (
            <div className="processed-video-section fallback">
                <div className="section-header">
                    <h3>⚠️ Local Playback Available</h3>
                    <div className="warning-badge">
                        <span>✅ Processed</span>
                        <span>⚠️ Local Only</span>
                    </div>
                </div>

                <div className="fallback-notice">
                    <p>Video processing completed successfully, but cloud upload failed. Playing from local server.</p>
                </div>

                <div className="video-display">
                    <VideoPlayer
                        src={`http://localhost:8000/video/${processedVideo}`}
                        title={`Local: ${videoInfo.video_name}`}
                        className="main-player"
                    />
                </div>

                <div className="video-metadata">
                    <h4>📊 Processing Details</h4>
                    <div className="metadata-grid">
                        <div className="metadata-item">
                            <span className="metadata-label">Video Name:</span>
                            <span className="metadata-value">{videoInfo.video_name}</span>
                        </div>
                        <div className="metadata-item">
                            <span className="metadata-label">Analysis Mode:</span>
                            <span className="metadata-value">{videoInfo.processing_mode?.replace('_', ' ')}</span>
                        </div>
                        <div className="metadata-item">
                            <span className="metadata-label">Status:</span>
                            <span className="metadata-value warning">⚠️ Local Fallback</span>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return null;
};

export default ProcessedVideoDisplay;
