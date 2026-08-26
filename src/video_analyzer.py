import cv2


class VideoAnalyzer:

    def __init__(self, video_path):
        self.video_path = video_path

    def get_metadata(self):

        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        fps = video.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            video.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = total_frames / fps if fps > 0 else 0

        video.release()

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration
        }

        return metadata