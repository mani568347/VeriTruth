def calculate_score(prob, source_score, similarity_score):
    score = (
        prob * 0.5 +
        (source_score / 100) * 0.3 +
        similarity_score * 0.2
    ) * 100

    return round(score, 2)