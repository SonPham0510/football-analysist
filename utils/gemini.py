import os
from typing import Optional, Dict, Any
import google.generativeai as genai
import json


def analyze_video(video_url: str) -> str:
    """Use the Gemini API to analyze a football video with structured output for web display.

    Args:
        video_url: Direct URL to the processed video (e.g. Cloudinary URL)
        prompt: Optional extra instructions for the model.

    Returns:
        Generated analysis text in structured format for web display.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # ✅ Enhanced comprehensive prompt for web-friendly output
    full_prompt = f"""
    Analyze this football video in detail and provide a comprehensive tactical analysis: {video_url} and 
     analysis perfomance of player and get insight for coaches , focusing on tactical insights, strengths, weaknesses, and recommendations.

    Please structure your analysis as follows and use clear headings and bullet points for web display:

    ## MATCH OVERVIEW
    - Match phase analyzed (which minutes/periods)
    - Key events observed
    - Overall match tempo and intensity

    ## TEAM FORMATIONS & TACTICS

    ### Team A (specify color/side):
    - **Formation**: (e.g., 4-3-3, 4-4-2)
    - **Playing style**: (possession-based, counter-attacking, etc.)
    - **Key tactical patterns**:
      • Build-up play approach
      • Pressing triggers and intensity
      • Defensive shape and compactness

    ### Team B (specify color/side):
    - **Formation**: (e.g., 4-3-3, 4-4-2)
    - **Playing style**: (possession-based, counter-attacking, etc.)
    - **Key tactical patterns**:
      • Build-up play approach
      • Pressing triggers and intensity
      • Defensive shape and compactness

    ##  STRENGTHS ANALYSIS

    ### Team A Strengths:
    - **Attacking strengths**: (specific examples from video)
    - **Defensive strengths**: (specific examples from video)
    - **Individual performances**: (standout players and their roles)

    ### Team B Strengths:
    - **Attacking strengths**: (specific examples from video)
    - **Defensive strengths**: (specific examples from video)
    - **Individual performances**: (standout players and their roles)

    ##  WEAKNESSES & VULNERABILITIES

    ### Team A Weaknesses:
    - **Defensive vulnerabilities**: (where they can be exploited)
    - **Attacking limitations**: (what's not working well)
    - **Tactical gaps**: (specific positioning or movement issues)

    ### Team B Weaknesses:
    - **Defensive vulnerabilities**: (where they can be exploited)
    - **Attacking limitations**: (what's not working well)
    - **Tactical gaps**: (specific positioning or movement issues)

    ##  KEY INSIGHTS & PATTERNS
    - **Ball possession trends**: Who dominates and in which areas
    - **Transition moments**: Quality of defensive/offensive transitions
    - **Set piece effectiveness**: Corners, free kicks, throw-ins
    - **Pressure moments**: How teams handle high-pressure situations

    ##  TACTICAL ADJUSTMENTS OBSERVED
    - Formation changes during the match
    - Substitution impacts
    - In-game tactical adaptations

    ##  STRATEGIC RECOMMENDATIONS

    ### For Team A:
    - **To improve**: Specific actionable suggestions
    - **To exploit**: Opponent's weaknesses to target
    - **Tactical tweaks**: Formation or style adjustments

    ### For Team B:
    - **To improve**: Specific actionable suggestions
    - **To exploit**: Opponent's weaknesses to target
    - **Tactical tweaks**: Formation or style adjustments

    ##  MATCH TURNING POINTS
    - Key moments that changed the game flow
    - Critical decision points
    - Momentum shifts and their causes

    ##  STATISTICAL OBSERVATIONS
    - Player movement patterns
    - Passing accuracy and direction trends
    - Defensive actions and positioning
    - Attacking third entries and final third play

    Please provide specific examples from the video to support each point and ensure the analysis is actionable for coaches and players. Use clear, concise language suitable for both technical and non-technical football audiences.

    Format the output with proper markdown headings, bullet points, and emojis for better web presentation. Important  write in Vietnamese language.
    """

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing video: {str(e)}"


def analyze_video_with_statistics(
    video_url: str, statistics_data: Dict[str, Any]
) -> str:
    """Use the Gemini API to analyze a football video with statistical data for enhanced insights.

    Args:
        video_url: Direct URL to the processed video (e.g. Cloudinary URL)
        statistics_data: Dictionary containing match statistics from CSV analysis

    Returns:
        Generated enhanced analysis text in structured format for web display.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Extract key statistics for analysis
    speed_stats = statistics_data.get("speed_stats", {})
    possession_stats = statistics_data.get("possession_stats", {})
    event_stats = statistics_data.get("event_stats", {})
    team_comparison = statistics_data.get("team_comparison", {})
    match_timeline = statistics_data.get("match_timeline", {})
    summary = statistics_data.get("summary", {})

    # Create statistical context for the AI
    stats_context = f"""
    MATCH STATISTICS SUMMARY:
    - Total Players: {summary.get("total_players", 0)}
    - Match Duration: {summary.get("match_duration", 0):.1f} seconds
    - Total Phases: {summary.get("total_phases", 0)}
    - Total Teams: {summary.get("total_teams", 0)}

    SPEED ANALYSIS:
    - Average Speed All Players: {speed_stats.get("overall_stats", {}).get("avg_speed_all_players", 0):.2f} m/s
    - Maximum Speed Recorded: {speed_stats.get("overall_stats", {}).get("max_speed_recorded", 0):.2f} m/s
    - Fastest Players: {list(speed_stats.get("fastest_players", {}).keys())[:5] if speed_stats.get("fastest_players") else []}

    POSSESSION ANALYSIS:
    - Team Possession Percentages: {possession_stats.get("team_possession_percentage", {})}
    - Total Possession Phases: {possession_stats.get("total_phases", 0)}
    - Average Phase Duration: {possession_stats.get("avg_phase_duration", 0):.2f} seconds

    EVENT ANALYSIS:
    - Top Passers: {list(event_stats.get("top_passers", {}).keys())[:5] if event_stats.get("top_passers") else []}
    - Overall Event Counts: {event_stats.get("overall_event_counts", {})}

    TEAM COMPARISON:
    - Team Statistics: {json.dumps(team_comparison, indent=2)}
    """

    # Enhanced comprehensive prompt with statistical data
    full_prompt = f"""
    Analyze this football video with detailed statistical data and provide a comprehensive tactical analysis: {video_url}

    Use the following statistical data to enhance your analysis:

    {stats_context}

    Please provide a comprehensive analysis that combines visual observations with statistical insights. Structure your analysis as follows:

    ## MATCH OVERVIEW & STATISTICAL CONTEXT
    - Match duration and intensity based on data
    - Key statistical highlights
    - Overall match tempo and player performance metrics

    ## TEAM FORMATIONS & TACTICAL ANALYSIS

    ### Team A (based on data):
    - **Formation**: (identify from video)
    - **Playing style**: (analyze based on possession stats and movement patterns)
    - **Key tactical patterns**:
      • Build-up play approach (analyze from possession data)
      • Pressing triggers and intensity (from speed and movement data)
      • Defensive shape and compactness

    ### Team B (based on data):
    - **Formation**: (identify from video)
    - **Playing style**: (analyze based on possession stats and movement patterns)
    - **Key tactical patterns**:
      • Build-up play approach (analyze from possession data)
      • Pressing triggers and intensity (from speed and movement data)
      • Defensive shape and compactness

    ## PERFORMANCE ANALYSIS WITH STATISTICS

    ### Speed & Movement Analysis:
    - **Fastest players**: {speed_stats.get("fastest_players", {})}
    - **Team speed patterns**: {speed_stats.get("team_stats", {})}
    - **Movement efficiency**: Analyze based on speed vs. effectiveness

    ### Possession & Ball Control:
    - **Possession distribution**: {possession_stats.get("team_possession_percentage", {})}
    - **Passing patterns**: {event_stats.get("top_passers", {})}
    - **Phase transitions**: {possession_stats.get("avg_phase_duration", 0):.2f}s average

    ## STRENGTHS ANALYSIS

    ### Team A Strengths:
    - **Attacking strengths**: (specific examples from video + stats)
    - **Defensive strengths**: (specific examples from video + stats)
    - **Individual performances**: (standout players and their roles)

    ### Team B Strengths:
    - **Attacking strengths**: (specific examples from video + stats)
    - **Defensive strengths**: (specific examples from video + stats)
    - **Individual performances**: (standout players and their roles)

    ## WEAKNESSES & VULNERABILITIES

    ### Team A Weaknesses:
    - **Defensive vulnerabilities**: (where they can be exploited)
    - **Attacking limitations**: (what's not working well)
    - **Tactical gaps**: (specific positioning or movement issues)

    ### Team B Weaknesses:
    - **Defensive vulnerabilities**: (where they can be exploited)
    - **Attacking limitations**: (what's not working well)
    - **Tactical gaps**: (specific positioning or movement issues)

    ## KEY INSIGHTS & PATTERNS
    - **Ball possession trends**: Who dominates and in which areas
    - **Transition moments**: Quality of defensive/offensive transitions
    - **Set piece effectiveness**: Corners, free kicks, throw-ins
    - **Pressure moments**: How teams handle high-pressure situations

    ## TACTICAL ADJUSTMENTS OBSERVED
    - Formation changes during the match
    - Substitution impacts
    - In-game tactical adaptations

    ## STRATEGIC RECOMMENDATIONS

    ### For Team A:
    - **To improve**: Specific actionable suggestions based on stats
    - **To exploit**: Opponent's weaknesses to target
    - **Tactical tweaks**: Formation or style adjustments

    ### For Team B:
    - **To improve**: Specific actionable suggestions based on stats
    - **To exploit**: Opponent's weaknesses to target
    - **Tactical tweaks**: Formation or style adjustments

    ## MATCH TURNING POINTS
    - Key moments that changed the game flow
    - Critical decision points
    - Momentum shifts and their causes

    ## STATISTICAL OBSERVATIONS
    - Player movement patterns (from speed data)
    - Passing accuracy and direction trends (from event data)
    - Defensive actions and positioning
    - Attacking third entries and final third play

    Please provide specific examples from the video to support each point and ensure the analysis is actionable for coaches and players. Use clear, concise language suitable for both technical and non-technical football audiences.

    Format the output with proper markdown headings, bullet points, and emojis for better web presentation. Important: write in Vietnamese language.
    """

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing video with statistics: {str(e)}"
