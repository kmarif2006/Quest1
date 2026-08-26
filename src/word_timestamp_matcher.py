import re


class WordTimestampMatcher:

    def normalize(self, text):

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    def find_target_start(
        self,
        segments,
        target_dialogue
    ):

        target_words = (
            self.normalize(
                target_dialogue
            )
            .split()
        )

        if not target_words:

            return None

        all_words = []

        for segment in segments:

            words = segment.get(
                "words",
                []
            )

            for word_info in words:

                word = self.normalize(
                    word_info["word"]
                )

                if not word:

                    continue

                all_words.append(
                    {
                        "word": word,
                        "start": word_info["start"],
                        "end": word_info["end"]
                    }
                )

        target_length = len(
            target_words
        )

        for index in range(
            len(all_words) -
            target_length +
            1
        ):

            candidate_words = []

            for offset in range(
                target_length
            ):

                candidate_words.append(
                    all_words[
                        index + offset
                    ]["word"]
                )

            if candidate_words == target_words:

                return {

                    "start": all_words[
                        index
                    ]["start"],

                    "end": all_words[
                        index +
                        target_length -
                        1
                    ]["end"],

                    "text": " ".join(
                        candidate_words
                    )
                }

        return None