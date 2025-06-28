import React, { useState } from 'react';
import VideoService from '../../services/videoService';
import FileUpload from '../../components/FileUpload';
import VideoPreview from '../../components/VideoPreview';
import LoadingState from '../../components/LoadingState';
import ProcessedVideoDisplay from '../../components/ProcessedVideoDisplay';
import ErrorMessage from '../../components/ErrorMessage';
import "./index.css";

const Solution = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedMode, setSelectedMode] = useState('PLAYER_DETECTION');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedVideo, setProcessedVideo] = useState('');
  const [videoInfo, setVideoInfo] = useState(null);
  const [error, setError] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [loadingState, setLoadingState] = useState('');

  const modes = [
    'PITCH_DETECTION',
    'PLAYER_DETECTION',
    'BALL_DETECTION',
    'PLAYER_TRACKING',
    'TEAM_CLASSIFICATION',
    'JERSEY_DETECTION',
    'RADAR'
  ];

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setProcessedVideo('');
    setVideoInfo(null);
    setError('');
    setLoadingState('');

    // Create preview URL for the selected video
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl('');
    }
  };

  // Cleanup preview URL when component unmounts
  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file first.');
      return;
    }

    setIsProcessing(true);
    setError('');
    setVideoInfo(null);
    setLoadingState('Uploading and processing video...');

    try {
      const result = await VideoService.uploadAndProcessVideo(selectedFile, selectedMode);

      if (result.success && result.data.success) {
        // SUCCESS: Video processed and uploaded!
        setLoadingState('🎉 Processing and upload complete!');

        // Set video info directly from response
        setVideoInfo({
          video_name: result.data.processed_video,
          cloudinary: result.data.cloudinary,
          processing_mode: result.data.processing_mode,
          file_size: result.data.file_size,
          status: {
            processing_complete: true,
            cloud_upload_complete: true,
            ready_to_view: true
          }
        });

        setProcessedVideo(result.data.processed_video);

        // Hide loading and show video after 1.5s
        setTimeout(() => {
          setLoadingState('');
          setIsProcessing(false);
        }, 1500);

      } else {
        // Processing failed
        setError(result.error || result.data?.message || 'Processing failed');
        setIsProcessing(false);
        setLoadingState('');
      }

    } catch (err) {
      console.error('Upload failed:', err);
      setError('Upload failed. Please check the server and try again.');
      setLoadingState('');
      setIsProcessing(false);
    }
  };

  return (
    <div className="solution-page">
      <div className="solution-header">
        <h1>⚽ Tactical Radar Generator</h1>
        <p className="solution-description">
          Upload your football match video and let our AI generate detailed tactical analysis with radar visualization.
        </p>
      </div>

      <FileUpload
        selectedFile={selectedFile}
        selectedMode={selectedMode}
        modes={modes}
        onFileChange={handleFileChange}
        onModeChange={setSelectedMode}
        onUpload={handleUpload}
        isProcessing={isProcessing}
      />

      <VideoPreview
        previewUrl={previewUrl}
        selectedFile={selectedFile}
      />

      <LoadingState
        loadingState={loadingState}
        isProcessing={isProcessing}
      />

      <ErrorMessage
        error={error}
        onDismiss={() => setError('')}
      />

      {/* Only show processed video when ready */}
      {processedVideo && !isProcessing && !loadingState && (
        <ProcessedVideoDisplay
          videoInfo={videoInfo}
          processedVideo={processedVideo}
        />
      )}
    </div>
  );
};

export default Solution;