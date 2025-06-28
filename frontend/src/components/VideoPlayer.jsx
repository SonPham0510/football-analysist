import React, { useRef, useState, useEffect } from 'react';
import './VideoPlayer.css';

const VideoPlayer = ({ src, title, className = '' }) => {
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [isMuted, setIsMuted] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const updateTime = () => setCurrentTime(video.currentTime);
        const updateDuration = () => setDuration(video.duration);
        const handleEnded = () => setIsPlaying(false);

        video.addEventListener('timeupdate', updateTime);
        video.addEventListener('loadedmetadata', updateDuration);
        video.addEventListener('ended', handleEnded);

        return () => {
            video.removeEventListener('timeupdate', updateTime);
            video.removeEventListener('loadedmetadata', updateDuration);
            video.removeEventListener('ended', handleEnded);
        };
    }, []);

    const togglePlay = () => {
        const video = videoRef.current;
        if (isPlaying) {
            video.pause();
        } else {
            video.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleSeek = (e) => {
        const video = videoRef.current;
        const rect = e.currentTarget.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        video.currentTime = pos * duration;
    };

    const handleVolumeChange = (e) => {
        const newVolume = parseFloat(e.target.value);
        setVolume(newVolume);
        videoRef.current.volume = newVolume;
        setIsMuted(newVolume === 0);
    };

    const toggleMute = () => {
        const video = videoRef.current;
        if (isMuted) {
            video.volume = volume;
            setIsMuted(false);
        } else {
            video.volume = 0;
            setIsMuted(true);
        }
    };

    const toggleFullscreen = () => {
        const video = videoRef.current;
        if (!isFullscreen) {
            if (video.requestFullscreen) {
                video.requestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
        setIsFullscreen(!isFullscreen);
    };

    const formatTime = (time) => {
        if (isNaN(time)) return '0:00';
        const minutes = Math.floor(time / 60);
        const seconds = Math.floor(time % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    return (
        <div className={`video-player ${className}`}>
            <div className="video-container">
                <video
                    ref={videoRef}
                    src={src}
                    className="video-element"
                    onClick={togglePlay}
                />

                <div className="video-controls">
                    <div className="progress-container">
                        <div
                            className="progress-bar"
                            onClick={handleSeek}
                        >
                            <div
                                className="progress-fill"
                                style={{ width: `${(currentTime / duration) * 100}%` }}
                            />
                        </div>
                    </div>

                    <div className="controls-row">
                        <div className="controls-left">
                            <button
                                className="control-btn play-btn"
                                onClick={togglePlay}
                            >
                                {isPlaying ? '⏸️' : '▶️'}
                            </button>

                            <button
                                className="control-btn volume-btn"
                                onClick={toggleMute}
                            >
                                {isMuted ? '🔇' : '🔊'}
                            </button>

                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={isMuted ? 0 : volume}
                                onChange={handleVolumeChange}
                                className="volume-slider"
                            />

                            <span className="time-display">
                                {formatTime(currentTime)} / {formatTime(duration)}
                            </span>
                        </div>

                        <div className="controls-right">
                            <button
                                className="control-btn fullscreen-btn"
                                onClick={toggleFullscreen}
                            >
                                ⛶
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {title && <p className="video-title">{title}</p>}
        </div>
    );
};

export default VideoPlayer;
