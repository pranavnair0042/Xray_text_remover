import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "dataset/processed"
MODEL_SAVE_PATH = "xray_inpaint_model.keras"

def load_and_preprocess(clean_path, corrupted_path, mask_path):
    clean = tf.io.decode_png(tf.io.read_file(clean_path), channels=1)
    corrupted = tf.io.decode_png(tf.io.read_file(corrupted_path), channels=1)
    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)

    clean = tf.cast(clean, tf.float32) / 255.0
    corrupted = tf.cast(corrupted, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    clean.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    corrupted.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    mask.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])

    inputs = tf.concat([corrupted, mask], axis=-1)
    return inputs, clean

def build_dataset():
    corrupted_files = sorted(glob.glob(os.path.join(DATA_DIR, "corrupted", "*.png")))
    clean_files = sorted(glob.glob(os.path.join(DATA_DIR, "clean", "*.png")))
    mask_files = sorted(glob.glob(os.path.join(DATA_DIR, "masks", "*.png")))

    dataset = tf.data.Dataset.from_tensor_slices((clean_files, corrupted_files, mask_files))
    dataset = dataset.shuffle(buffer_size=len(corrupted_files))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    val_size = max(1, int(len(corrupted_files) * 0.15))
    val_ds = dataset.take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = dataset.skip(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# Mask-weighted Loss: penalizes errors heavily inside the masked annotation area
class MaskedInpaintLoss(keras.losses.Loss):
    def __init__(self, mask_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_weight = mask_weight

    def call(self, y_true, y_pred):
        # Extract binary mask from channel 1 of the input inside the graph
        pass

def masked_mae_loss(y_true, y_pred, mask, mask_weight=8.0):
    diff = tf.abs(y_true - y_pred)
    unmasked_loss = diff * (1.0 - mask)
    masked_loss = diff * mask * mask_weight
    return tf.reduce_mean(unmasked_loss + masked_loss)

def build_lightweight_unet():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 2), name="model_input")
    corrupted_ch = inputs[:, :, :, 0:1]
    mask_ch = inputs[:, :, :, 1:2]

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    concat4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    concat5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Custom training step that allows accessing the mask channel for loss calculation
    class InpaintModel(keras.Model):
        def train_step(self, data):
            x, y = data
            mask = x[:, :, :, 1:2]
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = masked_mae_loss(y, y_pred, mask)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return {"loss": loss}

        def test_step(self, data):
            x, y = data
            mask = x[:, :, :, 1:2]
            y_pred = self(x, training=False)
            loss = masked_mae_loss(y, y_pred, mask)
            return {"loss": loss}

    wrapped_model = InpaintModel(inputs=inputs, outputs=outputs)
    return wrapped_model

if __name__ == "__main__":
    train_ds, val_ds = build_dataset()
    model = build_lightweight_unet()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    print("Training U-Net model...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_SAVE_PATH)
    print("Training complete and model saved.")