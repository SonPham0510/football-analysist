import React from 'react';
import './LoadingState.css';

const LoadingState = ({ loadingState, isProcessing }) => {
    if (!isProcessing && !loadingState) return null;

    const isComplete = loadingState.includes('complete');

    return (
        <div className="loading-section">
            <div className="loading-container">
                <div className="loading-spinner-container">
                    <div className={`loading-spinner ${isComplete ? 'complete' : ''}`}>
                        {isComplete ? '✅' : ''}
                    </div>
                </div>

                <p className="loading-text">
                    {loadingState || 'Processing your video...'}
                </p>

                <div className="loading-steps">
                    <div className="step completed">
                        <div className="step-icon">✅</div>
                        <span>Upload video to server</span>
                    </div>
                    <div className={`step ${!isComplete ? 'active' : 'completed'}`}>
                        <div className="step-icon">
                            {isComplete ? '✅' : '🤖'}
                        </div>
                        <span>AI video analysis & cloud upload</span>
                    </div>
                    <div className={`step ${isComplete ? 'completed' : ''}`}>
                        <div className="step-icon">
                            {isComplete ? '✅' : '⭕'}
                        </div>
                        <span>Ready to view!</span>
                    </div>
                </div>

                <div className="progress-bar">
                    <div
                        className="progress-fill"
                        style={{
                            width: isComplete ? '100%' : '65%'
                        }}
                    ></div>
                </div>

                <p className="loading-note">
                    ⏱️ Processing time varies based on video length and complexity
                </p>
            </div>
        </div>
    );
};

export default LoadingState;
