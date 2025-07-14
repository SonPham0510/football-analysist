import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


class CloudinaryManager:
    @staticmethod
    def upload_video(
        file_path: str, public_id: str = None, folder: str = "football_analysis"
    ):
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
                format="mp4",
            )
            return {
                "success": True,
                "public_id": result["public_id"],
                "secure_url": result["secure_url"],
                "playback_url": result.get("playback_url"),
                "duration": result.get("duration"),
                "format": result.get("format"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

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

    @staticmethod
    def upload_csv(
        file_path: str, public_id: str = None, folder: str = "football_analysis/csv"
    ):
        """
        Upload CSV file to Cloudinary
        """
        try:
            result = cloudinary.uploader.upload(
                file_path,
                resource_type="raw",  # CSV là raw file
                public_id=public_id,
                folder=folder,
                overwrite=True,
                format="csv",
            )
            return {
                "success": True,
                "public_id": result["public_id"],
                "secure_url": result["secure_url"],
                "url": result["url"],
                "bytes": result.get("bytes"),
                "format": result.get("format"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_csv_url(public_id: str):
        """
        Get CSV file URL from Cloudinary
        """
        try:
            url = cloudinary.utils.cloudinary_url(public_id, resource_type="raw")[0]
            return url
        except Exception as e:
            return None

    @staticmethod
    def download_csv(public_id: str, local_path: str):
        """
        Download CSV file from Cloudinary to local path
        """
        try:
            import requests

            url = CloudinaryManager.get_csv_url(public_id)
            if url:
                response = requests.get(url)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(response.content)
                return True
            return False
        except Exception as e:
            print(f"Error downloading CSV: {e}")
            return False

    @staticmethod
    def upload_match_data(video_path: str, csv_path: str, match_name: str):
        """
        Upload both video and CSV for a complete match
        """
        try:
            results = {}

            # Upload video
            video_result = CloudinaryManager.upload_video(
                video_path,
                public_id=f"{match_name}_video",
                folder="football_analysis/videos",
            )
            results["video"] = video_result

            # Upload CSV if exists
            if os.path.exists(csv_path):
                csv_result = CloudinaryManager.upload_csv(
                    csv_path,
                    public_id=f"{match_name}_data",
                    folder="football_analysis/csv",
                )
                results["csv"] = csv_result
            else:
                results["csv"] = {"success": False, "error": "CSV file not found"}

            return {
                "success": video_result["success"] and results["csv"]["success"],
                "video": video_result,
                "csv": results["csv"],
                "match_name": match_name,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
