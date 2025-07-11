import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import './MatchStatistics.css';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const MatchStatistics = ({ csvFileName, onClose }) => {
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);

  useEffect(() => {
    fetchStatistics();
  }, [csvFileName]);

  const fetchStatistics = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/statistics/${csvFileName}`);
      const data = await response.json();
      
      if (data.success) {
        setStatistics(data.data);
      } else {
        setError('Failed to load statistics');
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchHeatmapData = async (playerId = null, teamId = null) => {
    try {
      const params = new URLSearchParams();
      if (playerId) params.append('player_id', playerId);
      if (teamId) params.append('team_id', teamId);
      
      const response = await fetch(`/api/statistics/${csvFileName}/heatmap?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setHeatmapData(data.data);
      }
    } catch (err) {
      console.error('Error fetching heatmap data:', err);
    }
  };

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
                  <span>Avg Speed: {stats.avg_speed} m/s</span>
                </div>
                <div className="team-stat">
                  <span>Max Speed: {stats.max_speed} m/s</span>
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

  const renderSpeedStats = () => {
    if (!statistics?.speed_stats) return null;

    const { fastest_players, slowest_players, overall_stats } = statistics.speed_stats;

    const speedData = Object.entries(fastest_players).slice(0, 10).map(([playerId, stats]) => ({
      player: `Player ${playerId}`,
      maxSpeed: stats.max,
      avgSpeed: stats.mean
    }));

    return (
      <div className="speed-stats-container">
        <div className="stats-summary">
          <h3>Speed Overview</h3>
          <div className="speed-cards">
            <div className="stat-card">
              <h4>Average Speed</h4>
              <p>{overall_stats.avg_speed_all_players} m/s</p>
            </div>
            <div className="stat-card">
              <h4>Max Speed Recorded</h4>
              <p>{overall_stats.max_speed_recorded} m/s</p>
            </div>
            <div className="stat-card">
              <h4>Total Players</h4>
              <p>{overall_stats.total_players}</p>
            </div>
          </div>
        </div>

        <div className="speed-chart">
          <h3>Top 10 Fastest Players</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={speedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="player" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="maxSpeed" fill="#8884d8" name="Max Speed (m/s)" />
              <Bar dataKey="avgSpeed" fill="#82ca9d" name="Avg Speed (m/s)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="player-lists">
          <div className="fastest-players">
            <h4>🏃‍♂️ Fastest Players</h4>
            <ul>
              {Object.entries(fastest_players).slice(0, 5).map(([playerId, stats]) => (
                <li key={playerId}>
                  Player {playerId}: {stats.max} m/s (avg: {stats.mean} m/s)
                </li>
              ))}
            </ul>
          </div>
          
          <div className="slowest-players">
            <h4>🐢 Slowest Average Speed</h4>
            <ul>
              {Object.entries(slowest_players).slice(0, 5).map(([playerId, stats]) => (
                <li key={playerId}>
                  Player {playerId}: {stats.mean} m/s (max: {stats.max} m/s)
                </li>
              ))}
            </ul>
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
            <select 
              value={selectedTeam || ''} 
              onChange={(e) => {
                setSelectedTeam(e.target.value || null);
                setSelectedPlayer(null);
                fetchHeatmapData(null, e.target.value || null);
              }}
            >
              <option value="">Select Team</option>
              <option value="0">Team 0</option>
              <option value="1">Team 1</option>
            </select>

            <select 
              value={selectedPlayer || ''} 
              onChange={(e) => {
                setSelectedPlayer(e.target.value || null);
                setSelectedTeam(null);
                fetchHeatmapData(e.target.value || null, null);
              }}
            >
              <option value="">Select Player</option>
              {statistics?.speed_stats?.detailed_player_stats && 
                Object.keys(statistics.speed_stats.detailed_player_stats).map(playerId => (
                  <option key={playerId} value={playerId}>Player {playerId}</option>
                ))
              }
            </select>

            <button onClick={() => {
              setSelectedPlayer(null);
              setSelectedTeam(null);
              fetchHeatmapData();
            }}>
              Show All Players
            </button>
          </div>
        </div>

        <div className="heatmap-visualization">
          {heatmapData ? (
            <div className="heatmap-info">
              <p>Total Positions: {heatmapData.total_positions}</p>
              <p>X Range: {heatmapData.x_range[0].toFixed(1)} - {heatmapData.x_range[1].toFixed(1)}</p>
              <p>Y Range: {heatmapData.y_range[0].toFixed(1)} - {heatmapData.y_range[1].toFixed(1)}</p>
              
              <div className="top-activity-zones">
                <h4>Top Activity Zones:</h4>
                <ul>
                  {heatmapData.top_activity_zones?.slice(0, 5).map((zone, idx) => (
                    <li key={idx}>
                      Position ({zone.x.toFixed(1)}, {zone.y.toFixed(1)}): {zone.frequency} occurrences
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <p>Select a player or team to view heatmap data</p>
          )}
        </div>
      </div>
    );
  };

  const renderTimeline = () => {
    if (!statistics?.match_timeline) return null;

    const { timeline } = statistics.match_timeline;

    const timelineData = timeline.map(segment => ({
      time: `${segment.start_time}s`,
      avgSpeed: segment.avg_speed,
      team0Possession: segment.possession_team_0,
      team1Possession: segment.possession_team_1,
      totalActions: segment.total_actions
    }));

    return (
      <div className="timeline-container">
        <h3>Match Timeline</h3>
        <div className="timeline-chart">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="avgSpeed" stroke="#8884d8" name="Avg Speed (m/s)" />
              <Line type="monotone" dataKey="totalActions" stroke="#82ca9d" name="Total Actions" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="possession-timeline">
          <h4>Possession Over Time</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="team0Possession" stackId="a" fill="#0088FE" name="Team 0" />
              <Bar dataKey="team1Possession" stackId="a" fill="#00C49F" name="Team 1" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  if (loading) return <div className="loading">Loading statistics...</div>;
  if (error) return <div className="error">Error: {error}</div>;

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
            className={selectedTab === 'speed' ? 'tab active' : 'tab'}
            onClick={() => setSelectedTab('speed')}
          >
            Speed Stats
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
          <button 
            className={selectedTab === 'timeline' ? 'tab active' : 'tab'}
            onClick={() => setSelectedTab('timeline')}
          >
            Timeline
          </button>
        </div>

        <div className="tab-content">
          {selectedTab === 'overview' && renderOverview()}
          {selectedTab === 'speed' && renderSpeedStats()}
          {selectedTab === 'possession' && renderPossessionStats()}
          {selectedTab === 'events' && renderEventStats()}
          {selectedTab === 'heatmap' && renderHeatmap()}
          {selectedTab === 'timeline' && renderTimeline()}
        </div>
      </div>
    </div>
  );
};

export default MatchStatistics;
