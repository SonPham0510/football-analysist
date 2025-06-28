import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

class VideoService {
    static async uploadAndProcessVideo(file, mode) {
        const formData = new FormData();
        formData.append('file', file);
        const params = new URLSearchParams({ mode });

        try {
            const response = await axios.post(
                `${API_BASE_URL}/upload?${params.toString()}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: 600000 // 10 minutes timeout
                }
            );

            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            let errorMessage = 'Failed. Please try again.';

            if (error.code === 'ECONNABORTED') {
                errorMessage = 'Processing timeout. The video may be too large or complex.';
            } else if (error.response && error.response.data) {
                errorMessage = error.response.data.message || errorMessage;
            }

            return {
                success: false,
                error: errorMessage
            };
        }
    }

    static async getVideoInfo(videoName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/video-info/${videoName}`);
            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.message || 'Failed to fetch video info'
            };
        }
    }

    static async listVideos() {
        try {
            const response = await axios.get(`${API_BASE_URL}/list-videos`);
            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.message || 'Failed to fetch videos'
            };
        }
    }
}

export default VideoService;
