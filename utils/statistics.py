import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """Convert pandas/numpy objects to JSON serializable format."""
    if obj is None:
        return None
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other object, try to convert to string as fallback
        try:
            return str(obj)
        except Exception:
            return None


class MatchStatistics:
    """
    Complete match statistics analyzer for football match data.
    Analyzes CSV data with columns: time, teamId, playerId, x, y, hasBall, phaseId, speed
    """

    def __init__(self, csv_file_path: str):
        """Initialize with CSV file path."""
        self.csv_file_path = csv_file_path
        self.df: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> None:
        """Load and validate CSV data."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            required_columns = [
                "time",
                "teamId",
                "playerId",
                "x",
                "y",
                "hasBall",
                "phaseId",
                "speed",
            ]

            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(
                    f"Missing required columns. Expected: {required_columns}"
                )

            logger.info(f"Loaded {len(self.df)} records from {self.csv_file_path}")

        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise

    def get_possession_stats(self) -> Dict[str, Any]:
        """Get comprehensive ball possession statistics."""
        try:
            if self.df is None or self.df.empty:
                return {"error": "No data available"}

            # Analyze possession phases
            phases_df = self._analyze_possession_phases()

            if phases_df.empty:
                return {"error": "No possession phases found"}

            # Player possession statistics
            player_possession_data = (
                self.df[self.df["hasBall"]]
                .groupby("playerId")
                .agg(
                    {
                        "time": "count",  # Total times with ball
                        "phaseId": "nunique",  # Number of possession phases
                    }
                )
            )
            player_possession_data.columns = ["ball_touches", "possession_phases"]
            player_possession = player_possession_data

            # Team possession statistics
            team_possession_data = (
                self.df[self.df["hasBall"]]
                .groupby("teamId")
                .agg({"time": "count", "phaseId": "nunique"})
            )
            team_possession_data.columns = ["ball_touches", "possession_phases"]
            team_possession = team_possession_data

            # Phase type statistics
            phase_stats = phases_df["event_type"].value_counts().to_dict()

            # Top players by possession - convert keys to strings
            top_possession_players_df = player_possession.nlargest(10, "ball_touches")
            top_possession_players = {}
            for idx, row in top_possession_players_df.iterrows():
                top_possession_players[str(idx)] = {
                    "ball_touches": int(row["ball_touches"]),
                    "possession_phases": int(row["possession_phases"]),
                }

            # Calculate possession percentages
            total_possession = self.df[self.df["hasBall"]].shape[0]
            team_possession_pct = {}
            for team_id in self.df["teamId"].unique():
                team_ball_count = self.df[
                    (self.df["hasBall"]) & (self.df["teamId"] == team_id)
                ].shape[0]
                team_possession_pct[str(team_id)] = (
                    round((team_ball_count / total_possession) * 100, 1)
                    if total_possession > 0
                    else 0
                )

            # Convert team and player possession dict keys to strings
            team_possession_dict = {
                str(k): v for k, v in team_possession.to_dict("index").items()
            }
            player_possession_dict = {
                str(k): v for k, v in player_possession.to_dict("index").items()
            }

            return {
                "phase_statistics": phase_stats,
                "team_possession": team_possession_dict,
                "team_possession_percentage": team_possession_pct,
                "player_possession": player_possession_dict,
                "top_possession_players": top_possession_players,
                "total_phases": int(len(phases_df)),
                "avg_phase_duration": round(float(phases_df["duration"].mean()), 2)
                if "duration" in phases_df.columns
                else 0,
            }

        except Exception as e:
            logger.error(f"Error calculating possession stats: {e}")
            return {}

    def _analyze_possession_phases(self) -> pd.DataFrame:
        """Analyze possession phases to determine pass/hold/turnover events."""
        try:
            if self.df is None or self.df.empty:
                return pd.DataFrame()

            df_possession = (
                self.df[self.df["hasBall"]].sort_values("time").reset_index(drop=True)
            )

            if df_possession.empty:
                return pd.DataFrame()

            # Group by phase ID
            phases = (
                df_possession.groupby("phaseId")
                .agg(
                    {"time": ["first", "last"], "playerId": "first", "teamId": "first"}
                )
                .reset_index()
            )

            phases.columns = ["phaseId", "start_time", "end_time", "playerId", "teamId"]
            phases["duration"] = phases["end_time"] - phases["start_time"]

            # Determine event type for each phase
            phases["next_playerId"] = phases["playerId"].shift(-1)
            phases["next_teamId"] = phases["teamId"].shift(-1)

            def classify_event(row):
                if pd.isna(row["next_playerId"]):
                    return "match_end"
                if row["playerId"] == row["next_playerId"]:
                    return "hold"
                if row["teamId"] == row["next_teamId"]:
                    return "pass"
                else:
                    return "turnover"

            phases["event_type"] = phases.apply(classify_event, axis=1)

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
            player_events = (
                phases_df.groupby(["playerId", "event_type"])
                .size()
                .unstack(fill_value=0)
            )

            # Top players for each event type - convert keys to strings
            top_passers = {}
            if "pass" in player_events.columns:
                top_passers_df = player_events.nlargest(10, "pass")
                top_passers = {
                    str(k): int(v) for k, v in top_passers_df["pass"].to_dict().items()
                }

            top_holders = {}
            if "hold" in player_events.columns:
                top_holders_df = player_events.nlargest(10, "hold")
                top_holders = {
                    str(k): int(v) for k, v in top_holders_df["hold"].to_dict().items()
                }

            top_turnovers = {}
            if "turnover" in player_events.columns:
                top_turnovers_df = player_events.nlargest(10, "turnover")
                top_turnovers = {
                    str(k): int(v)
                    for k, v in top_turnovers_df["turnover"].to_dict().items()
                }

            # Team event statistics - convert keys to strings
            team_events_df = (
                phases_df.groupby(["teamId", "event_type"]).size().unstack(fill_value=0)
            )
            team_events = {
                str(k): v for k, v in team_events_df.to_dict("index").items()
            }

            # Event efficiency (pass success rate, etc.) - convert keys to strings
            player_efficiency = {}
            for player_id in player_events.index:
                total_events = player_events.loc[player_id].sum()
                passes = player_events.loc[player_id].get("pass", 0)
                turnovers = player_events.loc[player_id].get("turnover", 0)

                if total_events > 0:
                    player_efficiency[str(player_id)] = {
                        "pass_success_rate": round(
                            (passes / (passes + turnovers)) * 100, 1
                        )
                        if (passes + turnovers) > 0
                        else 0,
                        "total_events": int(total_events),
                        "passes": int(passes),
                        "turnovers": int(turnovers),
                        "holds": int(player_events.loc[player_id].get("hold", 0)),
                    }

            return {
                "top_passers": top_passers,
                "top_holders": top_holders,
                "top_turnovers": top_turnovers,
                "team_events": team_events,
                "player_efficiency": player_efficiency,
                "overall_event_counts": phases_df["event_type"]
                .value_counts()
                .to_dict(),
            }

        except Exception as e:
            logger.error(f"Error calculating event stats: {e}")
            return {}

    def get_heatmap_data(
        self,
        player_id: Optional[Union[int, str]] = None,
        team_id: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """Get heatmap data for player positions."""
        try:
            if self.df is None or self.df.empty:
                return {"error": "No data available"}

            df_filtered = self.df.copy()

            # Filter by player or team if specified
            if player_id is not None:
                # Convert to int if string
                player_id = int(player_id) if isinstance(player_id, str) else player_id
                df_filtered = df_filtered[df_filtered["playerId"] == player_id]

                if df_filtered.empty:
                    return {"error": f"No data found for player {player_id}"}

            elif team_id is not None:
                # Convert to int if string
                team_id = int(team_id) if isinstance(team_id, str) else team_id
                df_filtered = df_filtered[df_filtered["teamId"] == team_id]

                if df_filtered.empty:
                    return {"error": f"No data found for team {team_id}"}

            # Create position data for heatmap with team information
            positions = df_filtered[["x", "y", "teamId", "playerId"]].values.tolist()

            # Calculate position frequency with team info
            position_counts = (
                df_filtered.groupby(["x", "y", "teamId"])
                .size()
                .reset_index(name="frequency")
            )

            # Get activity zones (most frequent positions) with team info
            top_positions = position_counts.nlargest(20, "frequency")

            # Create team-specific position data
            team_positions = {}
            for tid in df_filtered["teamId"].unique():
                team_data = df_filtered[df_filtered["teamId"] == tid]
                team_position_counts = (
                    team_data.groupby(["x", "y"]).size().reset_index(name="frequency")
                )
                team_positions[str(tid)] = {
                    "positions": team_data[["x", "y"]].values.tolist(),
                    "position_frequency": team_position_counts.to_dict("records"),
                    "player_count": int(team_data["playerId"].nunique()),
                    "total_positions": len(team_data),
                }

            result = {
                "positions": positions,
                "position_frequency": position_counts.to_dict("records"),
                "top_activity_zones": top_positions.to_dict("records"),
                "team_positions": team_positions,
                "total_positions": len(positions),
                "x_range": [
                    float(df_filtered["x"].min()),
                    float(df_filtered["x"].max()),
                ],
                "y_range": [
                    float(df_filtered["y"].min()),
                    float(df_filtered["y"].max()),
                ],
            }

            # Add player info if filtering by specific player
            if player_id is not None:
                player_data = df_filtered[df_filtered["playerId"] == player_id]
                if not player_data.empty:
                    result["player_info"] = {
                        "player_id": int(player_id),
                        "team_id": int(player_data["teamId"].iloc[0]),
                        "total_positions": len(player_data),
                        "time_range": [
                            float(player_data["time"].min()),
                            float(player_data["time"].max()),
                        ],
                    }

            return result

        except Exception as e:
            logger.error(f"Error generating heatmap data: {e}")
            return {"error": str(e)}

    def get_team_comparison(self) -> Dict[str, Any]:
        """Get comprehensive team comparison statistics."""
        try:
            if self.df is None or self.df.empty:
                return {"error": "No data available"}

            team_stats = {}

            for team_id in self.df["teamId"].unique():
                team_data = self.df[self.df["teamId"] == team_id]

                # Convert team_id to string to ensure JSON serialization
                team_key = str(team_id)

                # Basic stats
                team_stats[team_key] = {
                    "player_count": int(team_data["playerId"].nunique()),
                    "total_actions": int(len(team_data)),
                    "possession_time": int(len(team_data[team_data["hasBall"]])),
                    "active_time_span": round(
                        float(team_data["time"].max() - team_data["time"].min()), 2
                    ),
                }

            # Calculate possession percentage
            total_possession = self.df[self.df["hasBall"]].shape[0]
            for team_key in team_stats:
                team_stats[team_key]["possession_percentage"] = (
                    round(
                        (team_stats[team_key]["possession_time"] / total_possession)
                        * 100,
                        1,
                    )
                    if total_possession > 0
                    else 0
                )

            return team_stats

        except Exception as e:
            logger.error(f"Error calculating team comparison: {e}")
            return {}

    def get_complete_analysis(self) -> Dict[str, Any]:
        """Get complete match analysis combining all statistics."""
        try:
            if self.df is None or self.df.empty:
                return {"error": "No data available in CSV file"}

            analysis = {
                "possession_stats": self.get_possession_stats(),
                "event_stats": self.get_event_stats(),
                "team_comparison": self.get_team_comparison(),
                "summary": {
                    "total_records": int(len(self.df)),
                    "total_players": int(self.df["playerId"].nunique()),
                    "total_teams": int(self.df["teamId"].nunique()),
                    "match_duration": round(
                        float(self.df["time"].max() - self.df["time"].min()), 2
                    )
                    if len(self.df) > 0
                    else 0,
                    "total_phases": int(self.df["phaseId"].nunique())
                    if "phaseId" in self.df.columns
                    else 0,
                },
            }

            # Convert all data to JSON serializable format
            return convert_to_serializable(analysis)

        except Exception as e:
            logger.error(f"Error generating complete analysis: {e}")
            return {"error": str(e)}
