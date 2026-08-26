import os

from src.ocr_detector import OCRDetector


FRAMES_DIRECTORY = "output/frames"


def main():

    detector = OCRDetector()

    frame_files = sorted(
        os.listdir(
            FRAMES_DIRECTORY
        )
    )

    print()

    print(
        "Scanning extracted frames..."
    )

    print(
        "--------------------------"
    )

    for filename in frame_files:

        if not filename.endswith(
            ".png"
        ):

            continue

        image_path = os.path.join(
            FRAMES_DIRECTORY,
            filename
        )

        extracted_text = (
            detector.extract_text(
                image_path
            )
        )

        if extracted_text:

            print()

            print(
                f"Frame: {filename}"
            )

            for item in extracted_text:

                print(
                    f"Text: "
                    f"{item['text']}"
                )

                print(
                    f"Confidence: "
                    f"{item['confidence']:.2f}"
                )


if __name__ == "__main__":

    main()