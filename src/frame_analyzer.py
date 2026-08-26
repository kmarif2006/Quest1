import cv2
import numpy as np


class FrameAnalyzer:

    def __init__(self):
        pass


    def calculate_difference(
        self,
        frame1,
        frame2
    ):

        gray1 = cv2.cvtColor(
            frame1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            frame2,
            cv2.COLOR_BGR2GRAY
        )

        difference = cv2.absdiff(
            gray1,
            gray2
        )

        score = np.mean(
            difference
        )

        return float(score)


    def analyze_frame_sequence(
        self,
        video_path,
        start_frame,
        end_frame
    ):

        video = cv2.VideoCapture(
            video_path
        )

        if not video.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        success, previous_frame = video.read()

        if not success:

            video.release()

            raise RuntimeError(
                "Could not read starting frame"
            )

        results = []

        current_frame_number = (
            start_frame + 1
        )

        while (
            current_frame_number <= end_frame
        ):

            success, current_frame = video.read()

            if not success:

                break

            difference_score = (
                self.calculate_difference(
                    previous_frame,
                    current_frame
                )
            )

            results.append(
                {
                    "frame": current_frame_number,
                    "difference": difference_score
                }
            )

            previous_frame = current_frame

            current_frame_number += 1

        video.release()

        return results


    def find_significant_changes(
        self,
        results,
        threshold=15.0
    ):

        significant_changes = []

        for result in results:

            if result["difference"] >= threshold:

                significant_changes.append(
                    result
                )

        return significant_changes