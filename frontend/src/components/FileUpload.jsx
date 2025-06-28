import React from 'react';
import './FileUpload.css';

const FileUpload = ({
    selectedFile,
    selectedMode,
    modes,
    onFileChange,
    onModeChange,
    onUpload,
    isProcessing
}) => {
    return (
        <div className="upload-section">
            <div className="upload-group">
                <label className="file-input-label">
                    <input
                        type="file"
                        accept="video/mp4"
                        onChange={onFileChange}
                        className="file-input"
                    />
                    <span className="file-input-text">
                        {selectedFile ? selectedFile.name : 'Choose Video File (MP4)'}
                    </span>
                    <span className="file-input-icon">📁</span>
                </label>
            </div>

            <div className="upload-group">
                <label className="mode-label">Analysis Mode:</label>
                <select
                    value={selectedMode}
                    onChange={(e) => onModeChange(e.target.value)}
                    className="mode-select"
                >
                    {modes.map(mode => (
                        <option key={mode} value={mode}>
                            {mode.replace('_', ' ')}
                        </option>
                    ))}
                </select>
            </div>

            <button
                onClick={onUpload}
                disabled={isProcessing || !selectedFile}
                className={`upload-btn ${isProcessing ? 'processing' : ''}`}
            >
                {isProcessing ? (
                    <>
                        <span className="btn-spinner"></span>
                        Processing...
                    </>
                ) : (
                    <>
                        <span className="btn-icon">🚀</span>
                        Upload & Analyze
                    </>
                )}
            </button>
        </div>
    );
};

export default FileUpload;
