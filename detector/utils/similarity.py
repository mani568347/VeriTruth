from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_score(vec, stored_vectors):
    try:
        similarity = cosine_similarity(vec, stored_vectors)
        return 1 - similarity.max()
    except:
        return 0.5