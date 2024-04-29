# Standardization and Normalization Tools
def standardize(data, mean, std):
    return (data - mean)/std

def unstandardize(data, mean, std):
    return data * std + mean

def normalize(x, bounds):
    return (x - bounds.T[0]) / (bounds.T[1] - bounds.T[0])

def unnormalize(x, bounds):
    return x * (bounds.T[1] - bounds.T[0]) + bounds.T[0]