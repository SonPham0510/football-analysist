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
                    timeout: 1200000 // 20 minutes timeout
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
    static async analyzeVideo(videoName, csvFilePath = null) {
        try {
            const requestData = {
                video_name: videoName
            };

            // Add csv_file_path if provided
            if (csvFilePath) {
                requestData.csv_file_path = csvFilePath;
            }

            console.log('Sending AI analysis request:', requestData);

            const response = await axios.post(`${API_BASE_URL}/ai-analyze`, requestData);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('AI Analysis Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || error.response?.data?.error || 'AI analysis failed'
            };
        }
    }
    // Enhanced analysis with statistics
    static async analyzeEnhanced(videoName, csvFilePath = null) {
        try {
            const requestData = {
                video_name: videoName
            };

            if (csvFilePath) {
                requestData.csv_file_path = csvFilePath;
            }

            console.log('Sending Enhanced analysis request:', requestData);

            const response = await axios.post(
                `${API_BASE_URL}/analyze-enhanced`,
                requestData,
                {
                    timeout: 300000, // 5 minutes timeout for AI analysis
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }
            );

            console.log('Enhanced analysis response status:', response.status);
            console.log('Enhanced analysis response data:', response.data);

            if (response.data && response.data.success) {
                return { success: true, data: response.data };
            } else {
                return {
                    success: false,
                    error: response.data?.detail || 'Enhanced analysis returned no data'
                };
            }
        } catch (error) {
            console.error('Enhanced Analysis Error Details:', {
                message: error.message,
                response: error.response?.data,
                status: error.response?.status,
                config: error.config
            });

            if (error.code === 'ECONNABORTED') {
                return {
                    success: false,
                    error: 'Enhanced analysis timeout. The analysis may take longer than expected.'
                };
            }

            return {
                success: false,
                error: error.response?.data?.detail || error.response?.data?.error || error.message || 'Enhanced analysis failed'
            };
        }
    }

    // Get saved enhanced analysis
    static async getEnhancedAnalysis(jsonFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/analysis/${jsonFileName}`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Get Enhanced Analysis Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Failed to retrieve enhanced analysis'
            };
        }
    }

    // List all enhanced analyses
    static async listEnhancedAnalyses() {
        try {
            const response = await axios.get(`${API_BASE_URL}/analysis`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('List Enhanced Analyses Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Failed to list enhanced analyses'
            };
        }
    }

    // Get statistics
    static async getStatistics(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Statistics Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Statistics retrieval failed'
            };
        }
    }

    // Get speed statistics
    static async getSpeedStats(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}/speed`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Speed Stats Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Speed statistics retrieval failed'
            };
        }
    }

    // Get possession statistics
    static async getPossessionStats(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}/possession`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Possession Stats Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Possession statistics retrieval failed'
            };
        }
    }

    // Get event statistics
    static async getEventStats(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}/events`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Event Stats Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Event statistics retrieval failed'
            };
        }
    }

    // Get heatmap data
    static async getHeatmapData(csvFileName, playerId = null, teamId = null) {
        try {
            const params = new URLSearchParams();
            if (playerId) params.append('player_id', playerId);
            if (teamId) params.append('team_id', teamId);

            const url = `${API_BASE_URL}/statistics/${csvFileName}/heatmap${params.toString() ? '?' + params.toString() : ''}`;
            const response = await axios.get(url);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Heatmap Data Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Heatmap data retrieval failed'
            };
        }
    }

    // Get team comparison
    static async getTeamComparison(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}/teams`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Team Comparison Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Team comparison retrieval failed'
            };
        }
    }

    // Get match timeline
    static async getMatchTimeline(csvFileName) {
        try {
            const response = await axios.get(`${API_BASE_URL}/statistics/${csvFileName}/timeline`);
            return { success: true, data: response.data };
        } catch (error) {
            console.error('Match Timeline Error:', error.response?.data);
            return {
                success: false,
                error: error.response?.data?.detail || 'Match timeline retrieval failed'
            };
        }
    }
}

export default VideoService;
