SMALL = 1e-16

def logit(x):
    return (x + SMALL).log() - (1. - x + SMALL).log()