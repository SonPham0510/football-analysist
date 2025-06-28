import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "dtcjrnltu"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

class CloudinaryManager:
    @staticmethod
    def upload_video(file_path: str, public_id: str = None, folder: str = "football_analysis"):
        """
        Upload video to Cloudinary
        """
        try:
            result = cloudinary.uploader.upload(
                file_path,
                resource_type="video",
                public_id=public_id,
                folder=folder,
                overwrite=True,
                format="mp4"
            )
            return {
                "success": True,
                "public_id": result["public_id"],
                "secure_url": result["secure_url"],
                "playback_url": result.get("playback_url"),
                "duration": result.get("duration"),
                "format": result.get("format")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_video_url(public_id: str, transformation=None):
        """
        Get video URL from Cloudinary
        """
        try:
            if transformation:
                url = cloudinary.CloudinaryVideo(public_id).build_url(**transformation)
            else:
                url = cloudinary.CloudinaryVideo(public_id).build_url()
            return url
        except Exception as e:
            return None
    
    @staticmethod
    def delete_video(public_id: str):
        """
        Delete video from Cloudinary
        """
        try:
            result = cloudinary.uploader.destroy(public_id, resource_type="video")
            return result.get("result") == "ok"
        except Exception as e:
            return False
    
    @staticmethod
    def get_player_embed_url(public_id: str, width=640, height=360):
        """
        Get Cloudinary video player embed URL
        """
        cloud_name = cloudinary.config().cloud_name
        return f"https://player.cloudinary.com/embed/?cloud_name={cloud_name}&public_id={public_id}&profile=cld-default"
