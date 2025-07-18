import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import './MatchStatistics.css';
import { API_BASE_URL } from '../services/videoService';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const MatchStatistics = ({ csvFileName, onClose }) => {
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [filterType, setFilterType] = useState(null); // 'team', 'player', or null

  console.log('MatchStatistics component rendered with csvFileName:', csvFileName);

  useEffect(() => {
    console.log('MatchStatistics useEffect triggered for csvFileName:', csvFileName);
    fetchStatistics();
  }, [csvFileName]);

  const fetchStatistics = async () => {
    try {
      setLoading(true);
      setError(null);

      // Use the correct API endpoint (without /api prefix)
      const response = await fetch(`${API_BASE_URL}/statistics/${csvFileName}`);
      console.log('Fetching statistics from:', response.url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.success && data.data) {
        setStatistics(data.data);
        setError(null);
      } else {
        setError(data.detail || 'Failed to load statistics - no data available');
        setStatistics(null);
      }
    } catch (err) {
      console.error('Statistics fetch error:', err);
      setError(`Failed to load statistics: ${err.message}`);
      setStatistics(null);
    } finally {
      setLoading(false);
    }
  };

  const fetchHeatmapData = async (playerId = null, teamId = null) => {
    try {
      const params = new URLSearchParams();
      if (playerId) params.append('player_id', playerId);
      if (teamId) params.append('team_id', teamId);

      const response = await fetch(`${API_BASE_URL}/statistics/${csvFileName}/heatmap?${params}`);
      const data = await response.json();

      if (data.success) {
        setHeatmapData(data.data);
      } else {
        console.error('Heatmap error:', data.detail);
      }
    } catch (err) {
      console.error('Error fetching heatmap data:', err);
    }
  };

  // Handle loading state
  if (loading) {
    return (
      <div className="match-statistics-modal">
        <div className="modal-content">
          <div className="modal-header">
            <h2>Loading Statistics...</h2>
            <button className="close-button" onClick={onClose}>×</button>
          </div>
          <div className="loading-content">
            <div className="spinner"></div>
            <p>Analyzing match data...</p>
          </div>
        </div>
      </div>
    );
  }

  // Handle error state
  if (error) {
    return (
      <div className="match-statistics-modal">
        <div className="modal-content">
          <div className="modal-header">
            <h2>Error Loading Statistics</h2>
            <button className="close-button" onClick={onClose}>×</button>
          </div>
          <div className="error-content">
            <div className="error-icon">⚠️</div>
            <h3>Unable to Load Statistics</h3>
            <p className="error-message">{error}</p>
            <div className="error-suggestions">
              <h4>Possible solutions:</h4>
              <ul>
                <li>Make sure the video was processed with RADAR mode</li>
                <li>Check that the CSV file was generated successfully</li>
                <li>Try refreshing the page</li>
              </ul>
            </div>
            <button
              className="retry-button"
              onClick={() => {
                setError(null);
                fetchStatistics();
              }}
            >
              🔄 Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Handle no statistics state
  if (!statistics) {
    return (
      <div className="match-statistics-modal">
        <div className="modal-content">
          <div className="modal-header">
            <h2>No Statistics Available</h2>
            <button className="close-button" onClick={onClose}>×</button>
          </div>
          <div className="no-data-content">
            <div className="no-data-icon">📊</div>
            <h3>No Match Data Found</h3>
            <p>Statistics are not available for this video.</p>
            <p>Make sure the video was processed using <strong>RADAR</strong> mode to generate statistics.</p>
          </div>
        </div>
      </div>
    );
  }

  const renderOverview = () => {
    if (!statistics) return null;

    const { summary, team_comparison } = statistics;

    return (
      <div className="overview-container">
        <div className="summary-cards">
          <div className="stat-card">
            <h3>Match Duration</h3>
            <p className="stat-value">{summary.match_duration}s</p>
          </div>
          <div className="stat-card">
            <h3>Total Players</h3>
            <p className="stat-value">{summary.total_players}</p>
          </div>
          <div className="stat-card">
            <h3>Total Records</h3>
            <p className="stat-value">{summary.total_records.toLocaleString()}</p>
          </div>
          <div className="stat-card">
            <h3>Total Phases</h3>
            <p className="stat-value">{summary.total_phases}</p>
          </div>
        </div>

        <div className="team-comparison">
          <h3>Team Comparison</h3>
          <div className="team-stats">
            {Object.entries(team_comparison).map(([teamId, stats]) => (
              <div key={teamId} className="team-card">
                <h4>Team {teamId}</h4>
                <div className="team-stat">
                  <span>Players: {stats.player_count}</span>
                </div>
                <div className="team-stat">
                  <span>Possession: {stats.possession_percentage}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderPossessionStats = () => {
    if (!statistics?.possession_stats) return null;

    const { team_possession_percentage, top_possession_players, phase_statistics } = statistics.possession_stats;

    const possessionPieData = Object.entries(team_possession_percentage).map(([teamId, percentage]) => ({
      name: `Team ${teamId}`,
      value: percentage,
      fill: COLORS[parseInt(teamId)]
    }));

    const phaseData = Object.entries(phase_statistics).map(([event, count]) => ({
      event: event.charAt(0).toUpperCase() + event.slice(1),
      count
    }));

    return (
      <div className="possession-stats-container">
        <div className="possession-overview">
          <div className="possession-pie">
            <h3>Team Possession Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={possessionPieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {possessionPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="phase-distribution">
            <h3>Event Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={phaseData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="event" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="top-possession-players">
          <h3>🏆 Top Possession Players</h3>
          <div className="player-possession-list">
            {Object.entries(top_possession_players).slice(0, 10).map(([playerId, stats]) => (
              <div key={playerId} className="possession-player-card">
                <span className="player-name">Player {playerId}</span>
                <span className="ball-touches">{stats.ball_touches} touches</span>
                <span className="possession-phases">{stats.possession_phases} phases</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderEventStats = () => {
    if (!statistics?.event_stats) return null;

    const { top_passers, top_holders, top_turnovers, player_efficiency } = statistics.event_stats;

    return (
      <div className="event-stats-container">
        <div className="event-leaders">
          <div className="event-category">
            <h3>🎯 Top Passers</h3>
            <ul>
              {Object.entries(top_passers).slice(0, 5).map(([playerId, passes]) => (
                <li key={playerId}>
                  Player {playerId}: {passes} successful passes
                </li>
              ))}
            </ul>
          </div>

          <div className="event-category">
            <h3>🤝 Top Ball Holders</h3>
            <ul>
              {Object.entries(top_holders).slice(0, 5).map(([playerId, holds]) => (
                <li key={playerId}>
                  Player {playerId}: {holds} ball holds
                </li>
              ))}
            </ul>
          </div>

          <div className="event-category">
            <h3>⚠️ Most Turnovers</h3>
            <ul>
              {Object.entries(top_turnovers).slice(0, 5).map(([playerId, turnovers]) => (
                <li key={playerId}>
                  Player {playerId}: {turnovers} turnovers
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="player-efficiency">
          <h3>📊 Player Efficiency</h3>
          <div className="efficiency-table">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Passes</th>
                  <th>Turnovers</th>
                  <th>Success Rate</th>
                  <th>Total Events</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(player_efficiency)
                  .sort(([, a], [, b]) => b.pass_success_rate - a.pass_success_rate)
                  .slice(0, 10)
                  .map(([playerId, stats]) => (
                    <tr key={playerId}>
                      <td>Player {playerId}</td>
                      <td>{stats.passes}</td>
                      <td>{stats.turnovers}</td>
                      <td className={stats.pass_success_rate > 70 ? 'high-efficiency' : 'low-efficiency'}>
                        {stats.pass_success_rate}%
                      </td>
                      <td>{stats.total_events}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  const renderHeatmap = () => {
    return (
      <div className="heatmap-container">
        <div className="heatmap-controls">
          <h3>Player Heatmap</h3>
          <div className="heatmap-filters">
            <div className="filter-section">
              <label>Filter by Team:</label>
              <select
                value={selectedTeam || ''}
                disabled={filterType === 'player'}
                onChange={(e) => {
                  const teamValue = e.target.value || null;
                  setSelectedTeam(teamValue);
                  setSelectedPlayer(null);
                  setFilterType(teamValue ? 'team' : null);
                  fetchHeatmapData(null, teamValue);
                }}
              >
                <option value="">All Teams</option>
                <option value="0">Team 0</option>
                <option value="1">Team 1</option>
              </select>
            </div>

            <div className="filter-section">
              <label>Filter by Player:</label>
              <select
                value={selectedPlayer || ''}
                onChange={(e) => {
                  const playerValue = e.target.value || null;
                  setSelectedPlayer(playerValue);
                  setFilterType(playerValue ? 'player' : null);
                  fetchHeatmapData(playerValue, null);
                }}
              >
                <option value="">All Players</option>
                {statistics?.possession_stats?.top_possession_players &&
                  Object.keys(statistics.possession_stats.top_possession_players).map(playerId => (
                    <option key={playerId} value={playerId}>Player {playerId}</option>
                  ))
                }
                {/* Fallback: use event stats if possession stats not available */}
                {!statistics?.possession_stats?.top_possession_players &&
                  statistics?.event_stats?.top_passers &&
                  Object.keys(statistics.event_stats.top_passers).map(playerId => (
                    <option key={playerId} value={playerId}>Player {playerId}</option>
                  ))
                }
              </select>
            </div>

            <button
              className="clear-filters-btn"
              onClick={() => {
                setSelectedPlayer(null);
                setFilterType(null);
                fetchHeatmapData();
              }}
            >
              Clear All Filters
            </button>
          </div>

          {/* Filter Status Display */}
          <div className="filter-status">
            {filterType === 'player' && selectedPlayer && (
              <span className="active-filter">👤 Showing Player {selectedPlayer}</span>
            )}
            {!filterType && (
              <span className="active-filter">🌐 Showing All Players</span>
            )}
          </div>
        </div>

        <div className="heatmap-visualization">
          {heatmapData ? (
            <div className="heatmap-content">
              <div className="heatmap-info">
                <div className="heatmap-stats">
                  <div className="stat-item">
                    <span className="stat-label">Total Positions:</span>
                    <span className="stat-value">{heatmapData.total_positions}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">X Range:</span>
                    <span className="stat-value">{heatmapData.x_range[0].toFixed(1)} - {heatmapData.x_range[1].toFixed(1)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Y Range:</span>
                    <span className="stat-value">{heatmapData.y_range[0].toFixed(1)} - {heatmapData.y_range[1].toFixed(1)}</span>
                  </div>
                </div>
              </div>

              {/* Soccer Field Heatmap */}
              <div className="heatmap-chart">
                <h4>🏟️ Player Movement Heatmap</h4>
                <div className="soccer-field-heatmap">
                  {/* Field Lines and Areas -  */}
                  <div className="field-lines"></div>
                  <div className="center-spot"></div>
                  <div className="goal-area-left"></div>
                  <div className="goal-area-right"></div>
                  <div className="penalty-area-left"></div>
                  <div className="penalty-area-right"></div>
                  <div className="penalty-spot-left"></div>
                  <div className="penalty-spot-right"></div>

                  {/* Player Position Dots */}
                  {heatmapData.position_frequency && heatmapData.position_frequency.map((position, index) => {
                    const xPercent = ((position.x - heatmapData.x_range[0]) / (heatmapData.x_range[1] - heatmapData.x_range[0])) * 100;
                    const yPercent = ((position.y - heatmapData.y_range[0]) / (heatmapData.y_range[1] - heatmapData.y_range[0])) * 100;
                    return (
                      <div
                        key={index}
                        className="player-dot"
                        style={{
                          left: `${Math.max(1, Math.min(99, xPercent))}%`,
                          top: `${Math.max(1, Math.min(99, 100 - yPercent))}%`,
                          opacity: Math.min(1, 0.3 + (position.frequency / 50)),
                          width: `${Math.max(6, Math.min(16, 6 + position.frequency / 10))}px`,
                          height: `${Math.max(6, Math.min(16, 6 + position.frequency / 10))}px`
                        }}
                        title={`Position (${position.x.toFixed(1)}, ${position.y.toFixed(1)}) - Activity: ${position.frequency}`}
                      />
                    );
                  })}

                  {/* Heat Zones for high activity areas */}
                  {heatmapData.top_activity_zones && heatmapData.top_activity_zones.slice(0, 5).map((zone, index) => {
                    const xPercent = ((zone.x - heatmapData.x_range[0]) / (heatmapData.x_range[1] - heatmapData.x_range[0])) * 100;
                    const yPercent = ((zone.y - heatmapData.y_range[0]) / (heatmapData.y_range[1] - heatmapData.y_range[0])) * 100;

                    let heatClass = 'low';
                    if (zone.frequency > 50) heatClass = 'very-high';
                    else if (zone.frequency > 30) heatClass = 'high';
                    else if (zone.frequency > 15) heatClass = 'medium';

                    const zoneSize = Math.min(100, 20 + (zone.frequency / 2));

                    return (
                      <div
                        key={`zone-${index}`}
                        className={`heat-zone ${heatClass}`}
                        style={{
                          left: `${Math.max(0, Math.min(95, xPercent - zoneSize / 2))}%`,
                          top: `${Math.max(0, Math.min(95, yPercent - zoneSize / 2))}%`,
                          width: `${zoneSize}px`,
                          height: `${zoneSize}px`
                        }}
                      />
                    );
                  })}
                </div>

                {/* Heat Intensity Legend */}
                <div className="heat-legend">
                  <span className="heat-legend-label">Activity Level:</span>
                  <div className="heat-scale">
                    <div className="heat-scale-item low" title="Low Activity"></div>
                    <div className="heat-scale-item medium" title="Medium Activity"></div>
                    <div className="heat-scale-item high" title="High Activity"></div>
                    <div className="heat-scale-item very-high" title="Very High Activity"></div>
                  </div>
                </div>
              </div>

              {/* Top Activity Zones Table */}
              <div className="top-activity-zones">
                <h4>🔥 Hottest Activity Zones</h4>
                <div className="activity-zones-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Position (X, Y)</th>
                        <th>Activity Count</th>
                        <th>Intensity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {heatmapData.top_activity_zones?.slice(0, 10).map((zone, idx) => (
                        <tr key={idx} className={idx < 3 ? 'top-zone' : ''}>
                          <td>#{idx + 1}</td>
                          <td>({zone.x.toFixed(1)}, {zone.y.toFixed(1)})</td>
                          <td>{zone.frequency}</td>
                          <td>
                            <div className="intensity-bar">
                              <div
                                className="intensity-fill"
                                style={{
                                  width: `${(zone.frequency / heatmapData.top_activity_zones[0].frequency) * 100}%`,
                                  backgroundColor: idx < 3 ? '#ff4757' : idx < 6 ? '#ffa502' : '#2ed573'
                                }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-heatmap-data">
              <div className="no-data-icon">🗺️</div>
              <h4>No Heatmap Data</h4>
              <p>Select a player to view their movement heatmap</p>
              <button
                onClick={() => fetchHeatmapData()}
                className="load-all-button"
              >
                Load All Players Heatmap
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="match-statistics-modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Match Statistics - {csvFileName}</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="tabs">
          <button
            className={selectedTab === 'overview' ? 'tab active' : 'tab'}
            onClick={() => setSelectedTab('overview')}
          >
            Overview
          </button>
          <button
            className={selectedTab === 'possession' ? 'tab active' : 'tab'}
            onClick={() => setSelectedTab('possession')}
          >
            Possession
          </button>
          <button
            className={selectedTab === 'events' ? 'tab active' : 'tab'}
            onClick={() => setSelectedTab('events')}
          >
            Events
          </button>
          <button
            className={selectedTab === 'heatmap' ? 'tab active' : 'tab'}
            onClick={() => {
              setSelectedTab('heatmap');
              if (!heatmapData) fetchHeatmapData();
            }}
          >
            Heatmap
          </button>
        </div>

        <div className="tab-content">
          {selectedTab === 'overview' && renderOverview()}
          {selectedTab === 'possession' && renderPossessionStats()}
          {selectedTab === 'events' && renderEventStats()}
          {selectedTab === 'heatmap' && renderHeatmap()}
        </div>
      </div>
    </div>
  );
};

export default MatchStatistics;
