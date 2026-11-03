from ops import collapse_output

def predict(model, x):
    return collapse_output(model(x))
