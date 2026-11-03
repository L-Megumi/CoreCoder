def normalize(x, mean, std):
    return (x - mean) / (std ** 2)
