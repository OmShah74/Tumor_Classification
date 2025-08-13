import os
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from data_utils import load_data, plot_sample_images, split_data

def build_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    return Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

def hms_string(sec_elapsed):
    h = int(sec_elapsed / 3600)
    m = int((sec_elapsed % 3600) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"

# --- Main execution ---
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    augmented_path = os.path.join(base_dir, "Augmented")
    augmented_yes = os.path.join(augmented_path, "yes")
    augmented_no = os.path.join(augmented_path, "no")

    IMG_WIDTH, IMG_HEIGHT = 240, 240
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
    plot_sample_images(X, y)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    model = build_model(IMG_SHAPE)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    logs_dir = os.path.join(base_dir, "logs", f"run_{int(time.time())}")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.keras")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=logs_dir)

    # Training
    start_time = time.time()
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])
    print(f"Training Time: {hms_string(time.time() - start_time)}")

    # Fine-tuning
    start_time = time.time()
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])
    print(f"Fine-tuning Time: {hms_string(time.time() - start_time)}")
