def least_confidence_score(normalized_probs):
    return 1 - normalized_probs.max(axis=1)
