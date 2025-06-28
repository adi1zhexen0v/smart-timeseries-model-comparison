import os
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError

from src.training.prepare_dataset import prepare_dataset

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

def build_lstm_model(input_shape, units=64, dropout=0.3, dense_units=32):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(1))
    model.summary()
    return model

def train_lstm(
    dataset_dir,
    output_dir,
    dataset_type="air_pollution",
    tag="default",
    target_column="PM2.5",
    sequence_length=30,
    epochs=100,
    batch_size=32,
    units=64,
    dropout=0.3,
    dense_units=32,
):
    prepared_dir = os.path.join(dataset_dir, "prepared_dataset")

    required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(prepared_dir, f))]

    if missing:
        logging.info("Preparing dataset...")
        input_path = os.path.join(dataset_dir, f"{dataset_type}_dataset.csv")
        feature_path = os.path.join(dataset_dir, "selected_features.json")
        prepare_dataset(
            input_path=input_path,
            features_path=feature_path,
            output_dir=prepared_dir,
            target_column=target_column,
            sequence_length=sequence_length,
        )

    data = load_dataset(prepared_dir)
    input_shape = data["X_train"].shape[1:]

    model = build_lstm_model(input_shape, units=units, dropout=dropout, dense_units=dense_units)
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

    os.makedirs(output_dir, exist_ok=True)
    model_name = f"{dataset_type}_{tag}_lstm.keras"
    save_path = os.path.join(output_dir, model_name)
    model.save(save_path)

    return model, history, data
