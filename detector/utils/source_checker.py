from urllib.parse import urlparse

# simple static credibility map (you can move to DB later)
SOURCE_SCORES = {
    "bbc.com": 95,
    "cnn.com": 90,
    "nytimes.com": 92
}

def get_source_score(url):
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return SOURCE_SCORES.get(domain, 50)
    except:
        return 50