from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A broad reference corpus of known-fake-news patterns, covering:
#   - Absurdist / satirical science claims
#   - Pseudoscience & miracle cures
#   - Political conspiracies & hoaxes
#   - Celebrity fabrications
#   - Cosmic / supernatural / paranormal
#   - Medical misinformation
#   - Historical revisionism / fringe theories
FAKE_SAMPLES = [
    # Absurdist / satirical science
    "Astronomers confirm the moon is made of green cheese after cosmic alignment",
    "Scientists discover the sun is actually a giant light bulb installed by aliens",
    "NASA confirms Earth is flat after secret satellite footage leaked",
    "Researchers prove gravity does not exist and it was invented by the government",
    "Scientists say drinking bleach cures all known diseases instantly",
    "Study proves humans only use 10 percent of their brain and can unlock powers",
    "Physicists confirm time travel is available at local pharmacy",
    "Biologists discover mermaids living beneath the Pacific Ocean",
    "Geologists confirm the Earth is hollow and populated inside",

    # Cosmic / supernatural / paranormal
    "Rare cosmic alignment causes humans to develop telepathic abilities",
    "Astronomers confirm planet Nibiru will collide with Earth next month",
    "Government confirms aliens landed and are living among us",
    "Vatican secretly confirms extraterrestrial life controls world governments",
    "NASA hides proof that the moon landing was filmed in Hollywood studio",
    "Secret government project reveals moon is an artificial alien space station",
    "Solar eclipse causes permanent blindness in anyone who glances at sky",

    # Political conspiracies & hoaxes
    "Government puts mind control chemicals in drinking water supply",
    "5G towers confirmed to spread viruses and control human behavior",
    "Bill Gates secretly implanting microchips through vaccines worldwide",
    "Deep state planning to cancel elections and seize permanent control",
    "World leaders confirmed to be shape-shifting reptilian aliens",
    "Secret cabal of elites harvesting children in underground tunnels",
    "Government admits chemtrails are mind control experiments on citizens",
    "Election results manipulated by secret machines controlled by globalists",

    # Medical misinformation
    "Miracle cure for all diseases discovered and suppressed by Big Pharma",
    "Doctor reveals single herb that permanently cures cancer overnight",
    "Vaccines confirmed to cause autism in children according to leaked study",
    "Drinking hot water with lemon completely eliminates coronavirus infection",
    "New supplement reverses aging completely within seven days guaranteed",
    "Scientists confirm that common spice kills all cancer cells instantly",

    # Celebrity fabrications
    "Celebrity caught in shocking scandal involving secret criminal organization",
    "Famous actor secretly dead and replaced by government clone for years",
    "Pop star confirms she is an illuminati puppet controlled by handlers",
    "Billionaire tech CEO arrested for running secret underground laboratory",

    # Economic hoaxes
    "Central banks planning to secretly delete all personal savings accounts",
    "Cryptocurrency discovered that automatically doubles your investment daily",
    "New law will ban all cash transactions and force citizens into poverty",

    # Historical revisionism / fringe
    "Ancient pyramids confirmed to be power generators built by lost civilization",
    "Historical documents prove World War Two was staged by bankers",
    "Secret archives reveal moon does not actually exist and is a projection",
]


def get_similarity_score(text: str) -> float:
    """
    Return the maximum cosine similarity between *text* and the fake-news
    reference corpus.  A score close to 1.0 means the text closely matches
    known fake-news language; close to 0.0 means it is linguistically
    distant from that corpus (though that alone does not prove it is real).
    """
    if not text or not text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(FAKE_SAMPLES + [text])
        sim = cosine_similarity(vectors[-1], vectors[:-1])
        return float(sim.max())
    except Exception:
        return 0.0
