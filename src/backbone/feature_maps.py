def get_feature_maps(model, x):
    for layer in model.features:
        x = layer(x)
    return x
