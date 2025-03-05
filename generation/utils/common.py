import unicodedata


def normalize_text(text):
    """ Normalize text using NFKC and replace curly apostrophes with straight ones """
    text = unicodedata.normalize("NFKC", text)  
    text = text.replace("’", "'")  
    return text.lower()
