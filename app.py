import os
import shutil
import uuid
import uvicorn
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


app = FastAPI(title="Football Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for AI analysis request
class AnalyzeRequest(BaseModel):
    video_name: str
    csv_file_path: str = None


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
            device="cpu",
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
    mode: str = "PLAYER_DETECTION",
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
        # Check if video exists
        video_path = PROCESSED_DIR / request.video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Processed video not found")

        # Check if CSV exists
        csv_name = request.csv_file_path or request.video_name.replace(".mp4", ".csv")
        csv_path = PROCESSED_DIR / csv_name
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV statistics file not found")

        # Load statistics data
        stats = MatchStatistics(str(csv_path))
        statistics_data = stats.get_complete_analysis()

        # Get video URL (for cloud storage, use the URL; for local, use file path)
        video_url = str(video_path)

        # Enhanced AI analysis with statistics
        from utils.gemini import analyze_video_with_statistics

        enhanced_analysis = analyze_video_with_statistics(video_url, statistics_data)

        return {
            "success": True,
            "enhanced_analysis": enhanced_analysis,
            "statistics_summary": {
                "total_players": statistics_data.get("summary", {}).get(
                    "total_players", 0
                ),
                "match_duration": statistics_data.get("summary", {}).get(
                    "match_duration", 0
                ),
                "total_phases": statistics_data.get("summary", {}).get(
                    "total_phases", 0
                ),
                "teams_analyzed": statistics_data.get("summary", {}).get(
                    "total_teams", 0
                ),
            },
            "csv_file": csv_name,
            "video_file": request.video_name,
        }

    except Exception as e:
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


@app.get("/statistics/{csv_filename}/timeline")
async def get_match_timeline(csv_filename: str):
    """Get match timeline with statistics over time."""
    try:
        csv_path = PROCESSED_DIR / csv_filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")

        stats = MatchStatistics(str(csv_path))
        timeline = stats.get_match_timeline()

        return {"success": True, "data": timeline}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/")
def root():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
