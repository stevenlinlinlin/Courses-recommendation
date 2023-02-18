import numpy as np
from transformers import EvalPrediction


def apk(actual, predicted, k=50):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    #print(score)
    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=50):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return {'map@50': np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])}


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    #print(preds, type(preds))
    predicted = [np.argsort(pred)[::-1].tolist() for pred in preds]
    actual = [np.where(label == 1)[0].tolist() for label in p.label_ids]

    result = mapk(
        actual=actual,
        predicted=predicted)
    return result
