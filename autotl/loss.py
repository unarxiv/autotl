import tensorlayer as tl

def classification_loss(pred, target):
    labels = target.argmax(1)
    return tl.cost.cross_entropy(pred, labels)

def regression_loss(pred, target):
    return tl.cost.mean_squared_error(pred, target.float())