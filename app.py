
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import  JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import uvicorn
from main import analyse_video, Mode
import os
from utils.utils import VideoUtils

app = FastAPI(title="Football Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # Step 1: Process video with AI
        analyse_video(
            source_video_path=str(source),
            target_video_path=str(target),
            device="cpu",
            mode=mode,
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


# @app.get("/video/{video_name}")
# def get_video(video_name: str):
#     """
#     Get video from Cloudinary or local fallback
#     """
#     # Try to get from Cloudinary first
#     public_id = Path(video_name).stem
#     cloud_info = VideoUtils.get_cloud_video_info(public_id)

#     if cloud_info["success"] and cloud_info["direct_url"]:
#         # Redirect to Cloudinary URL
#         return JSONResponse(
#             {
#                 "type": "cloudinary",
#                 "direct_url": cloud_info["direct_url"],
#                 "player_url": cloud_info["player_url"],
#                 "public_id": cloud_info["public_id"],
#             }
#         )

#     # Fallback to local file
#     video_path = os.path.join("/home/sonpham/thesis/processed", video_name)
#     print(f"Fetching video from local: {video_path}")

#     if not os.path.exists(video_path):
#         print(f"Video not found: {video_path}")
#         return JSONResponse({"error": "Video not found"}, status_code=404)

#     if not os.path.isfile(video_path):
#         print(f"Path is not a file: {video_path}")
#         return JSONResponse({"error": "Path is not a file"}, status_code=400)

#     # Get file size for proper headers
#     file_size = os.path.getsize(video_path)

#     headers = {
#         "Accept-Ranges": "bytes",
#         "Content-Length": str(file_size),
#         "Content-Type": "video/mp4",
#         "Cache-Control": "no-cache",
#     }

#     print(f"Serving local video: {video_path}, size: {file_size}")
#     return StreamingResponse(
#         open(video_path, "rb"), media_type="video/mp4", headers=headers
#     )


# @app.get("/video-info/{video_name}")
# def get_video_info(video_name: str):
#     """
#     Get video information (Cloudinary URLs, local path, etc.)
#     """
#     public_id = Path(video_name).stem
#     cloud_info = VideoUtils.get_cloud_video_info(public_id)

#     local_path = os.path.join("/home/sonpham/thesis/processed", video_name)
#     local_exists = os.path.exists(local_path)

#     # Check if file is actually processed (not just created)
#     processing_complete = False
#     file_size = 0

#     if local_exists:
#         file_size = os.path.getsize(local_path)
#         # Consider processing complete if file size > 1MB (indicates actual video content)
#         processing_complete = file_size > 1024 * 1024  # 1MB threshold

#     return {
#         "video_name": video_name,
#         "public_id": public_id,
#         "cloudinary": cloud_info,
#         "local": {
#             "exists": local_exists,
#             "path": local_path if local_exists else None,
#             "size": file_size,
#             "processing_complete": processing_complete,
#         },
#         "status": {
#             "processing_complete": processing_complete,
#             "cloud_upload_complete": cloud_info.get("success", False),
#             "ready_to_view": processing_complete and cloud_info.get("success", False),
#         },
#     }


# @app.get("/list-videos")
# def list_videos():
#     """List all processed videos with Cloudinary info"""
#     video_dir = "/home/sonpham/thesis/processed"
#     videos = []

#     if os.path.exists(video_dir):
#         for filename in os.listdir(video_dir):
#             if filename.endswith(".mp4"):
#                 file_path = os.path.join(video_dir, filename)
#                 public_id = Path(filename).stem
#                 cloud_info = VideoUtils.get_cloud_video_info(public_id)

#                 videos.append(
#                     {
#                         "name": filename,
#                         "url": f"/video/{filename}",
#                         "info_url": f"/video-info/{filename}",
#                         "size": os.path.getsize(file_path),
#                         "full_path": file_path,
#                         "cloudinary": cloud_info,
#                     }
#                 )

#     return {"videos": videos, "total": len(videos)}


@app.get("/")
def root():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
