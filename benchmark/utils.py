
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)

    return (p0 - pe) / (1 - pe)
