from keybert import KeyBERT
import re

class HashtagGenerator:
    def __init__(self):
        self.keyword_model = KeyBERT(model='distilbert-base-nli-mean-tokens')

    def generate_hashtags(self, caption: str, object_labels: list) -> list:
        keywords = self.keyword_model.extract_keywords(caption, top_n=5)
        keywords = [kw[0] for kw in keywords]

        # Combine with object labels
        candidates = set(keywords + object_labels)

        # Filter and format
        hashtags = []
        for word in candidates:
            word = word.lower()
            word = re.sub(r'[^a-z0-9]', '', word)
            if word:
                hashtags.append(f"#{word}")

        return sorted(set(hashtags))[:10]
