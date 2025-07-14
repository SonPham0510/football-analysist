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
            console.log('Starting enhanced analysis for:', videoName);
            console.log('CSV file:', csvFileName);

            // Use VideoService instead of direct fetch
            const result = await VideoService.analyzeEnhanced(videoName, csvFileName);
            console.log('Enhanced analysis result:', result);

            if (result.success) {
                setEnhancedResult(result.data.enhanced_analysis || 'Analysis completed successfully');
            } else {
                setEnhancedResult(`Enhanced analysis failed: ${result.error || 'Unknown error'}`);
                console.error('Enhanced analysis error:', result.error);
            }
        } catch (error) {
            console.error('Enhanced analysis exception:', error);
            setEnhancedResult(`Error: ${error.message || 'Failed to connect to server'}`);
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
                    {enhancedAnalyzing ? (
                        <span>
                            🔄 <span className="loading-text">Enhanced Analyzing...</span>
                            <small style={{ display: 'block', fontSize: '11px', marginTop: '4px' }}>
                                This may take 1-2 minutes for AI analysis
                            </small>
                        </span>
                    ) : (
                        '📊 Enhanced Analysis with Statistics'
                    )}
                </button>

                <button
                    onClick={() => {
                        console.log('View Statistics button clicked');
                        console.log('Current showStatistics state:', showStatistics);
                        console.log('CSV filename:', csvFileName);
                        setShowStatistics(!showStatistics);
                    }}
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
                    <div className="result-content enhanced-content">
                        {enhancedResult.includes('**') || enhancedResult.includes('#') ? (
                            // If it contains markdown formatting, render as HTML
                            <div
                                dangerouslySetInnerHTML={{
                                    __html: enhancedResult
                                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                        .replace(/### (.*?)\n/g, '<h4>$1</h4>')
                                        .replace(/## (.*?)\n/g, '<h3>$1</h3>')
                                        .replace(/# (.*?)\n/g, '<h2>$1</h2>')
                                        .replace(/• (.*?)\n/g, '<li>$1</li>')
                                        .replace(/\n/g, '<br>')
                                }}
                            />
                        ) : (
                            // Otherwise render as plain text with line breaks
                            <div dangerouslySetInnerHTML={{ __html: enhancedResult.replace(/\n/g, '<br>') }} />
                        )}
                    </div>

                    {enhancedResult.includes('AI analysis temporarily unavailable') && (
                        <div className="fallback-notice">
                            <small>💡 <strong>Note:</strong> This analysis is based on statistical data only. AI video analysis will be available when the service is restored.</small>
                        </div>
                    )}
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

                {/* Add statistics modal here too */}
                {showStatistics && (
                    <MatchStatistics
                        csvFileName={csvFileName}
                        onClose={() => setShowStatistics(false)}
                    />
                )}
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

                {/* Add statistics modal here too */}
                {showStatistics && (
                    <MatchStatistics
                        csvFileName={csvFileName}
                        onClose={() => setShowStatistics(false)}
                    />
                )}
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
            {showStatistics && (
                <>
                    {console.log('Rendering MatchStatistics modal in fallback section with csvFileName:', csvFileName)}
                    <MatchStatistics
                        csvFileName={csvFileName}
                        onClose={() => setShowStatistics(false)}
                    />
                </>
            )}
        </div>
    );
}

export default ProcessedVideoDisplay;
