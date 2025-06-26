import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError

def load_file(name, dataset_dir):
    return np.load(os.path.join(dataset_dir, f"{name}.npy"))

def load_dataset(dataset_dir):
    return {
        "X_train": load_file("X_train", dataset_dir),
        "y_train": load_file("y_train", dataset_dir),
        "X_val": load_file("X_val", dataset_dir),
        "y_val": load_file("y_val", dataset_dir),
        "X_test": load_file("X_test", dataset_dir),
        "y_test": load_file("y_test", dataset_dir),
    }

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    return model

def train_lstm(dataset_dir, save_path, epochs=100, batch_size=32):
    data = load_dataset(dataset_dir)
    input_shape = data["X_train"].shape[1:]

    model = build_lstm_model(input_shape)
    model.compile(
        loss="mse",
        optimizer=Adam(0.001),
        metrics=[MeanAbsoluteError()]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(save_path)
    return model, history, data
