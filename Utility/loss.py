def spectral_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    # Calculate grad
    grad_HR = y_true[:, :, :, :-1] - y_true[:, :, :, 1:]
    grad_SR = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]

    grad_diff = grad_HR - grad_SR
    grad_diff_1 = grad_diff[:, :, :, :-1]
    grad_diff_2 = grad_diff[:, :, :, 1:]

    spec_loss = K.mean(0.5 * K.square(grad_diff_1) +
                       0.5 * K.square(grad_diff_2), axis=-1)

    return spec_loss

def combined_loss(y_true, y_pred, A=0.1):
    y_true = K.cast(y_true, y_pred.dtype)

    spec_loss = spectral_loss(y_true, y_pred)
    normal_mse = K.mean(K.square(y_pred - y_true), axis=-1)
    merged_loss = normal_mse + A*spec_loss

    return merged_loss

def hallucination_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    den = K.square(y_true + 1) + 0.001
    weights = tf.divide(0.001, den)-0.95
    weights = K.relu(weights)+1
    normal_mse = K.square(y_pred - y_true)
    weighted_loss = normal_mse * weights
    mean_weighted_loss = K.mean(weighted_loss, axis=-1)

    return mean_weighted_loss