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
    
def res_conv_block(inputs, filters):
    """ResU-Net Residual Block"""
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    shortcut = layers.Conv2D(filters, 1, padding="same")(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    
    res = layers.Add()([x, shortcut])
    res = layers.LeakyReLU(alpha=0.1)(res)
    return res
    
def attention_gate(skip_connection, gating_signal, inter_filters):
    """Attention U-Net Attention Gate"""
    # Gating signal
    g = layers.Conv2D(inter_filters, 1, padding="same")(gating_signal)
    g = layers.BatchNormalization()(g)
    
    # Skip connection
    s = layers.Conv2D(inter_filters, 1, padding="same")(skip_connection)
    s = layers.BatchNormalization()(s)
    
    combined = layers.Activation('relu')(layers.Add()([g, s]))
    
    psi = layers.Conv2D(1, 1, padding="same")(combined)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)
    
    return layers.Multiply()([skip_connection, psi])

def encoder_block(inputs, filters, model_type="unet"):
    if model_type == "resunet":
        s = res_conv_block(inputs, filters)
    else:
        s = conv_block(inputs, filters)
    p = layers.MaxPooling2D(2)(s)
    return s, p
    
def decoder_block(inputs, skip_features, filters, model_type="unet"):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    
    if model_type == "attention":
        skip_features = attention_gate(skip_features, x, filters // 2)
        
    x = layers.Concatenate()([x, skip_features])
    
    if model_type == "resunet":
        x = res_conv_block(x, filters)
    else:
        x = conv_block(x, filters)
    return x

def build_model(model_type="unet", input_shape=(128, 128, 4)):
    inputs = layers.Input(input_shape)

    # --- ENCODER ---
    s1, p1 = encoder_block(inputs, 64, model_type)
    s2, p2 = encoder_block(p1, 128, model_type)
    s3, p3 = encoder_block(p2, 256, model_type)

    # --- BOTTLENECK ---
    if model_type == "resunet":
        b1 = res_conv_block(p3, 512)
    else:
        b1 = conv_block(p3, 512)

    # --- DECODER ---
    d1 = decoder_block(b1, s3, 256, model_type)
    d2 = decoder_block(d1, s2, 128, model_type)
    d3 = decoder_block(d2, s1, 64, model_type)
    
    outputs = layers.Conv2D(3, 1, activation="sigmoid", name="final_output")(d3)

    model = Model(inputs, outputs, name=model_type)
    return model

# Metrics and Loss Functions

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