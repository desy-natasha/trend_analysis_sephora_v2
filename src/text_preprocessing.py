import re
import html
import emoji                                      
import contractions as contractions_lib           
from nltk.corpus import stopwords                 

STOPWORDS = set(stopwords.words("english"))

def convert_emoji(text, mode = "remove"):
    """Convert emoji to words or remove them."""
    if mode == "keep":
        return text

    if mode == "convert":
        # demojize: 😍 → ":heart_eyes:" , 💄 → ":lipstick:"
        text = emoji.demojize(text, delimiters=(" ", " "))

        # clean up underscores left by demojize  (heart_eyes → heart eyes)
        text = text.replace("_", " ")
        return text

    # mode == "remove"
    return emoji.replace_emoji(text, replace=" ")

def expand_contractions(text):
    """Expand contractions to their full forms."""

    # e.g. can't → cannot, I've → I have
    text = contractions_lib.fix(text)

    return text

def remove_stop_words(text: str) -> str:
    """Remove stopwords from text."""

    text = " ".join(w for w in text.split() if w.lower() not in STOPWORDS)
    
    return text

def text_preprocessing_pipeline(text, emoji_mode = "convert", remove_punctuation= False, remove_stopwords= False):
    """Text preprocessing pipeline for review text."""
    
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)                          # 1. HTML entities (&amp; → &, &lt; → <, etc.)
    text = re.sub(r"<[^>]+>", " ", text)                # 2. HTML tags (<b>bold</b> → bold)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # 3. URLs (http/https links)
    text = text.lower()                                 # 4. Lowercase
    text = expand_contractions(text)                    # 5. Expand contractions
    text = convert_emoji(text, emoji_mode)              # 6. Convert emoji (opt)
    text = re.sub(r'\b\d+\b', '', text)                 # 7. Remove numbers

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)            # 8. Remove punctuation (opt)

    if remove_stopwords:
        text = remove_stop_words(text)                  # 9. Remove stopwords (opt)

    text = re.sub(r"\s{2,}", " ", text).strip()         # 10. Collapse whitespace

    return text
