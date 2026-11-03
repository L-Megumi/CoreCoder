def predict(model, x):
    return model(x).transpose(0, 1)
