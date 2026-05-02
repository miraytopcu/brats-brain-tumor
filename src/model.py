import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, filters):
    
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def encoder_block(inputs, filters):
    
    s = conv_block(inputs, filters)
    p = layers.MaxPooling2D(2)(s)
    return s, p

def decoder_block(inputs, skip_features, filters):
    
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x

def build_model(model_type="unet", input_shape=(128, 128, 4)):
    inputs = layers.Input(input_shape)

    # --- ENCODER ---
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)

    # --- BOTTLENECK ---
    b1 = conv_block(p3, 512)

    # --- DECODER ---
    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)
    
    outputs = layers.Conv2D(3, 1, activation="sigmoid", name="final_output")(d3)

    model = Model(inputs, outputs, name=model_type)
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = alpha * tf.pow(1.0 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def hybrid_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)