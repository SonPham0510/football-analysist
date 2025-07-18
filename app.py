import os
import shutil
import uuid
import uvicorn
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from main import analyse_video, Mode
from utils.utils import VideoUtils
from utils.gemini import analyze_video as gemini_analyze
from utils.statistics import MatchStatistics
from utils.gemini import analyze_video_with_statistics

app = FastAPI(title="Football Analyst API")

origins = [
    "http://localhost",
    "http://localhost:3000",

    "http://127.0.0.1:3000",
    "http://127.0.0.1",
    "http://localhost:5173",  # Vite default
    "*",  # For development, allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for AI analysis request
class AnalyzeRequest(BaseModel):
    video_name: str
    csv_file_path: Optional[str] = None


UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Thread pool for processing videos
executor = ThreadPoolExecutor(max_workers=2)


def process_video_complete(source: Path, target: Path, mode: Mode) -> dict:
    """
    Complete video processing function that returns result after both
    video processing and cloud upload are finished
    """
    try:
        print(f"Starting video processing: {source} -> {target}")

        # Generate CSV file path for RADAR mode
        csv_file_path = None
        if mode == Mode.RADAR:
            csv_file_path = str(PROCESSED_DIR / f"{target.stem}.csv")
            print(f"CSV will be saved to: {csv_file_path}")

        # Step 1: Process video with AI
        analyse_video(
            source_video_path=str(source),
            target_video_path=str(target),
            device="cuda",
            mode=mode,
            csv_file_path=csv_file_path,
        )

        print(f"Video processing completed: {target}")

        # Step 2: Upload processed video to Cloudinary
        print(f"Starting cloud upload: {target}")
        upload_result = VideoUtils.upload_processed_video_to_cloud(
            local_path=str(target), video_name=target.name
        )

        if upload_result["success"]:
            print(f"Cloud upload successful: {upload_result['public_id']}")
            return {
                "success": True,
                "message": "Video processed and uploaded successfully",
                "local_path": str(target),
                "csv_file": csv_file_path
                if csv_file_path and os.path.exists(csv_file_path)
                else None,
                "cloudinary": {
                    "public_id": upload_result["public_id"],
                    "player_url": upload_result["player_url"],
                    "direct_url": upload_result["direct_url"],
                    "cloudinary_url": upload_result["cloudinary_url"],
                },
            }
        else:
            print(f"Cloud upload failed: {upload_result['error']}")
            return {
                "success": False,
                "message": "Video processed but cloud upload failed",
                "error": upload_result["error"],
                "local_path": str(target),
                "csv_file": csv_file_path
                if csv_file_path and os.path.exists(csv_file_path)
                else None,
            }

    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        return {"success": False, "message": "Video processing failed", "error": str(e)}


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    mode: str = "RADAR",
):
    """
    Synchronous upload endpoint that processes video and uploads to cloud
    Returns result only when everything is complete
    """
    try:
        # Save uploaded file
        temp_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        mode_enum = Mode[mode]
        output_path = (
            PROCESSED_DIR
            / f"{temp_path.stem}-{mode_enum.value.lower()}{temp_path.suffix}"
        )

        print(f"Processing video in thread: {output_path.name}")

        # Submit to thread pool and wait for completion
        future = executor.submit(
            process_video_complete, temp_path, output_path, mode_enum
        )

        # This will block until processing + upload is complete
        result = future.result()  # This waits for the thread to finish

        # Cleanup uploaded file
        try:
            os.remove(temp_path)
        except Exception as cleanup_error:
            print(f"Failed to cleanup uploaded file: {cleanup_error}")
            pass

        if result["success"]:
            return JSONResponse(
                {
                    "success": True,
                    "message": result["message"],
                    "processed_video": output_path.name,
                    "csv_file": result.get("csv_file"),
                    "cloudinary": result["cloudinary"],
                    "processing_mode": mode,
                    "file_size": os.path.getsize(output_path)
                    if os.path.exists(output_path)
                    else 0,
                }
            )
        else:
            return JSONResponse(
                {
                    "success": False,
                    "message": result["message"],
                    "error": result["error"],
                    "processed_video": output_path.name
                    if "local_path" in result
                    else None,
                    "csv_file": result.get("csv_file"),
                    "processing_mode": mode,
                },
                status_code=500,
            )

    except Exception as e:
        print(f"Upload endpoint error: {str(e)}")
        return JSONResponse(
            {"success": False, "message": "Upload failed", "error": str(e)},
            status_code=500,
        )


@app.post("/ai-analyze")
async def ai_analyze(request: AnalyzeRequest):
    """Run Gemini AI analysis on a processed video."""
    video_path = PROCESSED_DIR / request.video_name
    csv_file_path = PROCESSED_DIR / f"{Path(request.video_name).stem}.csv"
    if not video_path.exists() and not csv_file_path.exists():
        raise HTTPException(
            status_code=404, detail="Video not found or CSV file not found"
        )

    public_id = Path(request.video_name).stem

    cloud_info = VideoUtils.get_cloud_video_info(public_id)
    video_url = cloud_info.get("direct_url") if cloud_info.get("success") else None

    if not video_url:
        video_url = str(video_path)

    try:
        analysis = gemini_analyze(video_url)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/analyze-enhanced")
async def analyze_enhanced_with_statistics(request: AnalyzeRequest):
    """
    Enhanced AI analysis using both video and statistical data.
    This provides more detailed and data-driven insights.
    """
    try:
        print(f"Enhanced analysis request: {request.video_name}")

        # Check if video exists
        video_path = PROCESSED_DIR / request.video_name
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            raise HTTPException(status_code=404, detail="Processed video not found")

        # Check if CSV exists - try multiple naming patterns
        if request.csv_file_path:
            csv_name = request.csv_file_path
        else:
            # Try different CSV naming patterns
            base_name = request.video_name.replace(".mp4", "")

            # If video name already contains "radar", use as-is
            if "radar" in request.video_name.lower():
                csv_name = request.video_name.replace(".mp4", ".csv")
            else:
                csv_name = f"{base_name}.csv"

        csv_path = PROCESSED_DIR / csv_name

        # If the primary CSV doesn't exist, try finding any CSV with similar name
        if not csv_path.exists():
            base_name = request.video_name.replace(".mp4", "").replace("-radar", "")
            csv_files = list(PROCESSED_DIR.glob(f"*{base_name}*.csv"))

            if csv_files:
                csv_path = csv_files[0]  # Use the first matching CSV
                csv_name = csv_path.name
                print(f"Found matching CSV: {csv_name}")
            else:
                print(f"CSV not found: {csv_path}")
                print(f"Tried looking for: *{base_name}*.csv")
                available_csvs = list(PROCESSED_DIR.glob("*.csv"))
                print(f"Available CSV files: {[f.name for f in available_csvs]}")
                raise HTTPException(
                    status_code=404,
                    detail=f"CSV statistics file not found. Tried: {csv_name}",
                )

        print(f"Loading statistics from: {csv_path}")
        # Load statistics data
        stats = MatchStatistics(str(csv_path))
        statistics_data = stats.get_complete_analysis()

        if not statistics_data or statistics_data.get("error"):
            error_msg = (
                statistics_data.get("error", "Unknown error")
                if statistics_data
                else "No data returned"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to generate statistics: {error_msg}"
            )

        print(
            f"Statistics loaded successfully. Summary: {statistics_data.get('summary', {})}"
        )

        # Get video URL (for cloud storage, use the URL; for local, use file path)
        video_url = str(video_path)
        print(f"Analyzing video: {video_url}")

        # Try enhanced analysis with error handling - Make AI analysis optional
        enhanced_analysis = "Enhanced analysis with statistics:\n\n"

        try:
            ai_analysis = analyze_video_with_statistics(video_url, statistics_data)
            enhanced_analysis += ai_analysis
            print("AI analysis completed successfully")
        except Exception as ai_error:
            print(f"AI analysis error: {ai_error}")
            # Continue without AI analysis, use statistics only
            enhanced_analysis += (
                f"AI analysis temporarily unavailable ({str(ai_error)})\n\n"
            )
            enhanced_analysis += "STATISTICAL ANALYSIS:\n"
            enhanced_analysis += f"• Total Players: {statistics_data.get('summary', {}).get('total_players', 0)}\n"
            enhanced_analysis += f"• Match Duration: {statistics_data.get('summary', {}).get('match_duration', 0):.1f} seconds\n"
            enhanced_analysis += f"• Total Phases: {statistics_data.get('summary', {}).get('total_phases', 0)}\n"

            # Add key insights from statistics (speed temporarily disabled)
            # speed_stats = statistics_data.get('speed_stats', {}).get('overall_stats', {})
            # if speed_stats:
            #     enhanced_analysis += f"• Average Speed: {speed_stats.get('avg_speed_all_players', 0):.2f} m/s\n"
            #     enhanced_analysis += f"• Maximum Speed: {speed_stats.get('max_speed_recorded', 0):.2f} m/s\n"

            possession_stats = statistics_data.get("possession_stats", {})
            if possession_stats.get("team_possession_percentage"):
                enhanced_analysis += f"• Team Possession: {possession_stats['team_possession_percentage']}\n"

        # Prepare complete analysis data with serialization
        analysis_result = {
            "success": True,
            "enhanced_analysis": enhanced_analysis,
            "statistics_data": statistics_data,
            "statistics_summary": {
                "total_players": int(
                    statistics_data.get("summary", {}).get("total_players", 0)
                ),
                "match_duration": float(
                    statistics_data.get("summary", {}).get("match_duration", 0)
                ),
                "total_phases": int(
                    statistics_data.get("summary", {}).get("total_phases", 0)
                ),
                "teams_analyzed": int(
                    statistics_data.get("summary", {}).get("total_teams", 0)
                ),
            },
            "csv_file": csv_name,
            "video_file": request.video_name,
            "analysis_timestamp": str(datetime.now()),
        }

        # Convert to JSON serializable format
        from utils.statistics import convert_to_serializable

        analysis_result = convert_to_serializable(analysis_result)

        # Save analysis to JSON file
        json_filename = request.video_name.replace(".mp4", "_enhanced_analysis.json")
        json_path = PROCESSED_DIR / json_filename

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"Analysis saved to: {json_path}")
        except Exception as save_error:
            print(f"Warning: Could not save analysis to file: {save_error}")

        # Add JSON file path to response
        analysis_result["json_file"] = json_filename
        analysis_result["json_path"] = str(json_path)

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Enhanced analysis error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Enhanced analysis failed: {str(e)}"
        )


# Statistics endpoints
@app.get("/statistics/{csv_filename}")
async def get_match_statistics(csv_filename: str):
    """Get complete match statistics from CSV file."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        analysis = stats.get_complete_analysis()

        return {"success": True, "data": analysis, "csv_file": csv_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{csv_filename}/speed")
async def get_speed_statistics(csv_filename: str):
    """Get player speed statistics."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        speed_stats = stats.get_player_speed_stats()

        return {"success": True, "data": speed_stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{csv_filename}/possession")
async def get_possession_statistics(csv_filename: str):
    """Get ball possession statistics."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        possession_stats = stats.get_possession_stats()

        return {"success": True, "data": possession_stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{csv_filename}/events")
async def get_event_statistics(csv_filename: str):
    """Get pass/hold/turnover event statistics."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        event_stats = stats.get_event_stats()

        return {"success": True, "data": event_stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{csv_filename}/heatmap")
async def get_heatmap_data(
    csv_filename: str, player_id: Optional[int] = None, team_id: Optional[int] = None
):
    """Get heatmap data for player positions."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        heatmap_data = stats.get_heatmap_data(player_id=player_id, team_id=team_id)

        return {"success": True, "data": heatmap_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics/{csv_filename}/teams")
async def get_team_comparison(csv_filename: str):
    """Get team comparison statistics."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        team_stats = stats.get_team_comparison()

        return {"success": True, "data": team_stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/statistics")
async def get_statistics_by_video(request: AnalyzeRequest):
    """
    Get match statistics from video name (matches frontend expectation).
    """
    try:
        # Determine CSV filename from video name
        csv_name = request.csv_file_path or request.video_name.replace(".mp4", ".csv")
        csv_path = PROCESSED_DIR / csv_name

        if not csv_path.exists():
            raise HTTPException(
                status_code=404, detail=f"CSV statistics file not found: {csv_name}"
            )

        # Load and analyze statistics
        stats = MatchStatistics(str(csv_path))
        analysis = stats.get_complete_analysis()

        return {
            "success": True,
            "statistics": analysis,
            "csv_file": csv_name,
            "video_file": request.video_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate statistics: {str(e)}"
        )


@app.get("/analysis/{json_filename}")
async def get_enhanced_analysis(json_filename: str):
    """Get saved enhanced analysis from JSON file."""
    try:
        # Ensure the filename has the correct extension
        if not json_filename.endswith("_enhanced_analysis.json"):
            if json_filename.endswith(".json"):
                json_filename = json_filename.replace(
                    ".json", "_enhanced_analysis.json"
                )
            else:
                json_filename = json_filename + "_enhanced_analysis.json"

        json_path = PROCESSED_DIR / json_filename
        if not json_path.exists():
            raise HTTPException(
                status_code=404, detail="Enhanced analysis file not found"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

        return {
            "success": True,
            "data": analysis_data,
            "json_file": json_filename,
            "file_path": str(json_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load analysis: {str(e)}"
        )


@app.get("/analysis")
async def list_enhanced_analyses():
    """List all available enhanced analysis JSON files."""
    try:
        json_files = list(PROCESSED_DIR.glob("*_enhanced_analysis.json"))

        analyses = []
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                analyses.append(
                    {
                        "filename": json_file.name,
                        "video_file": data.get("video_file", ""),
                        "csv_file": data.get("csv_file", ""),
                        "timestamp": data.get("analysis_timestamp", ""),
                        "file_size": json_file.stat().st_size,
                    }
                )
            except Exception:
                # Skip corrupted files
                continue

        return {"success": True, "analyses": analyses, "total_count": len(analyses)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list analyses: {str(e)}"
        )


@app.get("/")
def root():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)
