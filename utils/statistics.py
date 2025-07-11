import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MatchStatistics:
    """
    Complete match statistics analyzer for football match data.
    Analyzes CSV data with columns: time, teamId, playerId, x, y, hasBall, phaseId, speed
    """

    def __init__(self, csv_file_path: str):
        """Initialize with CSV file path."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.load_data()

    def load_data(self) -> None:
        """Load and validate CSV data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            required_columns = ['time', 'teamId', 'playerId', 'x', 'y', 'hasBall', 'phaseId', 'speed']
            
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")
                
            logger.info(f"Loaded {len(self.df)} records from {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise

    def get_player_speed_stats(self) -> Dict[str, Any]:
        """Get comprehensive player speed statistics."""
        try:
            player_speeds = self.df.groupby('playerId')['speed'].agg([
                'count', 'mean', 'max', 'min', 'std'
            ]).round(2)
            
            # Top 10 fastest players
            fastest_players = player_speeds.nlargest(10, 'max')[['max', 'mean']].to_dict('index')
            
            # Top 10 slowest players (by average speed)
            slowest_players = player_speeds.nsmallest(10, 'mean')[['max', 'mean']].to_dict('index')
            
            # Overall speed statistics
            overall_stats = {
                'total_players': len(player_speeds),
                'avg_speed_all_players': round(self.df['speed'].mean(), 2),
                'max_speed_recorded': round(self.df['speed'].max(), 2),
                'min_speed_recorded': round(self.df['speed'].min(), 2),
                'speed_std': round(self.df['speed'].std(), 2)
            }
            
            # Speed by team
            team_speed_stats = self.df.groupby('teamId')['speed'].agg([
                'mean', 'max', 'count'
            ]).round(2).to_dict('index')
            
            return {
                'overall_stats': overall_stats,
                'fastest_players': fastest_players,
                'slowest_players': slowest_players,
                'team_stats': team_speed_stats,
                'detailed_player_stats': player_speeds.to_dict('index')
            }
            
        except Exception as e:
            logger.error(f"Error calculating speed stats: {e}")
            return {}

    def get_possession_stats(self) -> Dict[str, Any]:
        """Get comprehensive ball possession statistics."""
        try:
            # Analyze possession phases
            phases_df = self._analyze_possession_phases()
            
            if phases_df.empty:
                return {"error": "No possession phases found"}
            
            # Player possession statistics
            player_possession = self.df[self.df['hasBall'] == True].groupby('playerId').agg({
                'time': 'count',  # Total times with ball
                'phaseId': 'nunique'  # Number of possession phases
            }).rename(columns={'time': 'ball_touches', 'phaseId': 'possession_phases'})
            
            # Team possession statistics
            team_possession = self.df[self.df['hasBall'] == True].groupby('teamId').agg({
                'time': 'count',
                'phaseId': 'nunique'
            }).rename(columns={'time': 'ball_touches', 'phaseId': 'possession_phases'})
            
            # Phase type statistics
            phase_stats = phases_df['event_type'].value_counts().to_dict()
            
            # Top players by possession
            top_possession_players = player_possession.nlargest(10, 'ball_touches').to_dict('index')
            
            # Calculate possession percentages
            total_possession = self.df[self.df['hasBall'] == True].shape[0]
            team_possession_pct = {}
            for team_id in self.df['teamId'].unique():
                team_ball_count = self.df[(self.df['hasBall'] == True) & (self.df['teamId'] == team_id)].shape[0]
                team_possession_pct[team_id] = round((team_ball_count / total_possession) * 100, 1) if total_possession > 0 else 0
            
            return {
                'phase_statistics': phase_stats,
                'team_possession': team_possession.to_dict('index'),
                'team_possession_percentage': team_possession_pct,
                'player_possession': player_possession.to_dict('index'),
                'top_possession_players': top_possession_players,
                'total_phases': len(phases_df),
                'avg_phase_duration': round(phases_df['duration'].mean(), 2) if 'duration' in phases_df.columns else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating possession stats: {e}")
            return {}

    def _analyze_possession_phases(self) -> pd.DataFrame:
        """Analyze possession phases to determine pass/hold/turnover events."""
        try:
            df_possession = self.df[self.df['hasBall'] == True].sort_values('time').reset_index(drop=True)
            
            if df_possession.empty:
                return pd.DataFrame()
            
            # Group by phase ID
            phases = df_possession.groupby('phaseId').agg({
                'time': ['first', 'last'],
                'playerId': 'first',
                'teamId': 'first'
            }).reset_index()
            
            phases.columns = ['phaseId', 'start_time', 'end_time', 'playerId', 'teamId']
            phases['duration'] = phases['end_time'] - phases['start_time']
            
            # Determine event type for each phase
            phases['next_playerId'] = phases['playerId'].shift(-1)
            phases['next_teamId'] = phases['teamId'].shift(-1)
            
            def classify_event(row):
                if pd.isna(row['next_playerId']):
                    return 'match_end'
                if row['playerId'] == row['next_playerId']:
                    return 'hold'
                if row['teamId'] == row['next_teamId']:
                    return 'pass'
                else:
                    return 'turnover'
            
            phases['event_type'] = phases.apply(classify_event, axis=1)
            
            return phases
            
        except Exception as e:
            logger.error(f"Error analyzing possession phases: {e}")
            return pd.DataFrame()

    def get_event_stats(self) -> Dict[str, Any]:
        """Get pass, hold, turnover statistics per player."""
        try:
            phases_df = self._analyze_possession_phases()
            
            if phases_df.empty:
                return {"error": "No possession phases found"}
            
            # Count events by player
            player_events = phases_df.groupby(['playerId', 'event_type']).size().unstack(fill_value=0)
            
            # Top players for each event type
            top_passers = player_events.nlargest(10, 'pass')['pass'].to_dict() if 'pass' in player_events.columns else {}
            top_holders = player_events.nlargest(10, 'hold')['hold'].to_dict() if 'hold' in player_events.columns else {}
            top_turnovers = player_events.nlargest(10, 'turnover')['turnover'].to_dict() if 'turnover' in player_events.columns else {}
            
            # Team event statistics
            team_events = phases_df.groupby(['teamId', 'event_type']).size().unstack(fill_value=0)
            
            # Event efficiency (pass success rate, etc.)
            player_efficiency = {}
            for player_id in player_events.index:
                total_events = player_events.loc[player_id].sum()
                passes = player_events.loc[player_id].get('pass', 0)
                turnovers = player_events.loc[player_id].get('turnover', 0)
                
                if total_events > 0:
                    player_efficiency[player_id] = {
                        'pass_success_rate': round((passes / (passes + turnovers)) * 100, 1) if (passes + turnovers) > 0 else 0,
                        'total_events': int(total_events),
                        'passes': int(passes),
                        'turnovers': int(turnovers),
                        'holds': int(player_events.loc[player_id].get('hold', 0))
                    }
            
            return {
                'top_passers': top_passers,
                'top_holders': top_holders,
                'top_turnovers': top_turnovers,
                'team_events': team_events.to_dict('index'),
                'player_efficiency': player_efficiency,
                'overall_event_counts': phases_df['event_type'].value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error calculating event stats: {e}")
            return {}

    def get_heatmap_data(self, player_id: Optional[int] = None, team_id: Optional[int] = None) -> Dict[str, Any]:
        """Get heatmap data for playersheatmap cua cau thu or teams."""
        try:
            df_filtered = self.df.copy()
            
            # Filter by player or team if specified
            if player_id is not None:
                df_filtered = df_filtered[df_filtered['playerId'] == player_id]
            elif team_id is not None:
                df_filtered = df_filtered[df_filtered['teamId'] == team_id]
            
            if df_filtered.empty:
                return {"error": "No data found for specified filters"}
            
            # Create position data for heatmap
            positions = df_filtered[['x', 'y']].values.tolist()
            
            # Calculate position frequency
            position_counts = df_filtered.groupby(['x', 'y']).size().reset_index(name='frequency')
            
            # Get activity zones (most frequent positions)
            top_positions = position_counts.nlargest(20, 'frequency')
            
            return {
                'positions': positions,
                'position_frequency': position_counts.to_dict('records'),
                'top_activity_zones': top_positions.to_dict('records'),
                'total_positions': len(positions),
                'x_range': [float(df_filtered['x'].min()), float(df_filtered['x'].max())],
                'y_range': [float(df_filtered['y'].min()), float(df_filtered['y'].max())]
            }
            
        except Exception as e:
            logger.error(f"Error generating heatmap data: {e}")
            return {}

    def get_team_comparison(self) -> Dict[str, Any]:
        """Get comprehensive team comparison statistics."""
        try:
            team_stats = {}
            
            for team_id in self.df['teamId'].unique():
                team_data = self.df[self.df['teamId'] == team_id]
                
                # Basic stats
                team_stats[team_id] = {
                    'player_count': team_data['playerId'].nunique(),
                    'total_actions': len(team_data),
                    'avg_speed': round(team_data['speed'].mean(), 2),
                    'max_speed': round(team_data['speed'].max(), 2),
                    'possession_time': len(team_data[team_data['hasBall'] == True]),
                    'active_time_span': round(team_data['time'].max() - team_data['time'].min(), 2)
                }
            
            # Calculate possession percentage
            total_possession = self.df[self.df['hasBall'] == True].shape[0]
            for team_id in team_stats:
                team_stats[team_id]['possession_percentage'] = round(
                    (team_stats[team_id]['possession_time'] / total_possession) * 100, 1
                ) if total_possession > 0 else 0
            
            return team_stats
            
        except Exception as e:
            logger.error(f"Error calculating team comparison: {e}")
            return {}

    def get_match_timeline(self) -> Dict[str, Any]:
        """Get match timeline with key events and statistics over time."""
        try:
            # Divide match into time segments (e.g., every 30 seconds)
            time_segments = []
            min_time = self.df['time'].min()
            max_time = self.df['time'].max()
            segment_duration = 30  # seconds
            
            current_time = min_time
            while current_time < max_time:
                end_time = min(current_time + segment_duration, max_time)
                
                segment_data = self.df[
                    (self.df['time'] >= current_time) & (self.df['time'] < end_time)
                ]
                
                if not segment_data.empty:
                    # Calculate stats for this segment
                    possession_by_team = segment_data[segment_data['hasBall'] == True]['teamId'].value_counts()
                    avg_speed = segment_data['speed'].mean()
                    
                    time_segments.append({
                        'start_time': round(current_time, 1),
                        'end_time': round(end_time, 1),
                        'avg_speed': round(avg_speed, 2),
                        'possession_team_0': int(possession_by_team.get(0, 0)),
                        'possession_team_1': int(possession_by_team.get(1, 0)),
                        'total_actions': len(segment_data)
                    })
                
                current_time = end_time
            
            return {
                'timeline': time_segments,
                'total_duration': round(max_time - min_time, 2),
                'segment_count': len(time_segments)
            }
            
        except Exception as e:
            logger.error(f"Error generating match timeline: {e}")
            return {}

    def get_complete_analysis(self) -> Dict[str, Any]:
        """Get complete match analysis combining all statistics."""
        try:
            return {
                'speed_stats': self.get_player_speed_stats(),
                'possession_stats': self.get_possession_stats(),
                'event_stats': self.get_event_stats(),
                'team_comparison': self.get_team_comparison(),
                'match_timeline': self.get_match_timeline(),
                'summary': {
                    'total_records': len(self.df),
                    'total_players': self.df['playerId'].nunique(),
                    'total_teams': self.df['teamId'].nunique(),
                    'match_duration': round(self.df['time'].max() - self.df['time'].min(), 2),
                    'total_phases': self.df['phaseId'].nunique()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating complete analysis: {e}")
            return {"error": str(e)}
