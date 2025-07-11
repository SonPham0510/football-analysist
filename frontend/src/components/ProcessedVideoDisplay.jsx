import React, { useState } from 'react';
import VideoPlayer from './VideoPlayer';
import MatchStatistics from './MatchStatistics';
import VideoService from '../services/videoService';
import './ProcessedVideoDisplay.css';

const ProcessedVideoDisplay = ({ videoInfo, processedVideo }) => {
    const [analyzing, setAnalyzing] = useState(false);
    const [aiResult, setAiResult] = useState('');
    const [showStatistics, setShowStatistics] = useState(false);
    const [enhancedAnalyzing, setEnhancedAnalyzing] = useState(false);
    const [enhancedResult, setEnhancedResult] = useState('');
    
    if (!videoInfo || !processedVideo) return null;

    const isCloudReady = videoInfo.status?.ready_to_view && videoInfo.cloudinary;
    const isLocalFallback = videoInfo.status?.processing_complete && !videoInfo.status?.cloud_upload_complete;
    const isRadarMode = videoInfo.processing_mode === 'RADAR';
    
    // Generate CSV filename from video name
    const csvFileName = processedVideo.replace('.mp4', '.csv');

    const runAnalysis = async () => {
        setAnalyzing(true);
        setAiResult('');
        const videoName = processedVideo || videoInfo.processed_video || videoInfo.video_name;
        const result = await VideoService.analyzeVideo(videoName);
        if (result.success) {
            setAiResult(result.data.analysis);
        } else {
            setAiResult(result.error || 'Analysis failed');
        }
        setAnalyzing(false);
    };

    const runEnhancedAnalysis = async () => {
        setEnhancedAnalyzing(true);
        setEnhancedResult('');
        const videoName = processedVideo || videoInfo.processed_video || videoInfo.video_name;
        
        try {
            const response = await fetch('/api/analyze-enhanced', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_name: videoName,
                    csv_file_path: csvFileName
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                setEnhancedResult(result.enhanced_analysis);
            } else {
                setEnhancedResult(result.detail || 'Enhanced analysis failed');
            }
        } catch (error) {
            setEnhancedResult('Error: ' + error.message);
        } finally {
            setEnhancedAnalyzing(false);
        }
    };

    const renderAISection = () => (
        <div className="ai-section">
            <h4>🤖 AI Analysis Options</h4>
            
            <div className="ai-buttons">
                <button onClick={runAnalysis} disabled={analyzing} className="ai-btn">
                    {analyzing ? '🔄 Analyzing Video...' : '📹 Basic Video Analysis'}
                </button>
                
                <button onClick={runEnhancedAnalysis} disabled={enhancedAnalyzing} className="ai-btn enhanced">
                    {enhancedAnalyzing ? '🔄 Enhanced Analyzing...' : '📊 Enhanced Analysis with Statistics'}
                </button>
                
                <button 
                    onClick={() => setShowStatistics(!showStatistics)} 
                    className="ai-btn stats"
                >
                    {showStatistics ? '📈 Hide Statistics' : '📈 View Match Statistics'}
                </button>
            </div>

            {aiResult && (
                <div className="analysis-result">
                    <h5>📹 Basic Video Analysis:</h5>
                    <div className="result-content" dangerouslySetInnerHTML={{ __html: aiResult.replace(/\n/g, '<br>') }} />
                </div>
            )}
            
            {enhancedResult && (
                <div className="analysis-result enhanced">
                    <h5>🚀 Enhanced Analysis with Statistics:</h5>
                    <div className="result-content" dangerouslySetInnerHTML={{ __html: enhancedResult.replace(/\n/g, '<br>') }} />
                </div>
            )}
        </div>
    );

    // If cloud is ready, show cloud video
    if (isCloudReady) {
        return (
            <div className="processed-video-section success">
                <div className="section-header">
                    <h3> Analysis Complete!</h3>
                    <div className="success-badge">
                        <span> Processed</span>
                        <span>Cloud Ready</span>
                    </div>
                </div>

                <div className="video-display">
                    <VideoPlayer
                        src={videoInfo.cloudinary.direct_url}
                        title={`Processed: ${videoInfo.video_name}`}
                        className="main-player"
                    />
                </div>
                {isRadarMode && renderAISection()}
            </div>
        );
    }

    // If only local processing is complete, show local video
    if (isLocalFallback) {
        return (
            <div className="processed-video-section fallback">
                <div className="section-header">
                    <h3> Processing Complete </h3>
                    <div className="warning-badge">
                        <span> Processed</span>
                        <span>Local Only</span>
                    </div>
                </div>

                <div className="video-display">
                    <VideoPlayer
                        src={`http://localhost:8000/video/${processedVideo}`}
                        title={`Processed: ${processedVideo}`}
                        className="main-player"
                    />
                </div>
                {isRadarMode && renderAISection()}
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
                    src={videoInfo.cloudinary?.direct_url || `http://localhost:8000/video/${processedVideo}`}
                    title={`Processed: ${videoInfo.video_name || processedVideo}`}
                    className="main-player"
                />
            </div>
            {isRadarMode && renderAISection()}
            {showStatistics && <MatchStatistics videoName={processedVideo} />}
        </div>
    );
}

export default ProcessedVideoDisplay;
