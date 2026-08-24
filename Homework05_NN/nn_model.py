# nn_model.py
import json
import os
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, hidden_layers=[256, 128, 64], dropout_rate=0.3):
    """รองรับการปรับเปลี่ยนจำนวน hidden layers และ neurons ผ่าน hidden_layers list"""
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.Flatten())

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                hidden_layers=[256, 128, 64], output_dir=None, epochs=30, batch_size=32):
    """Build, train and save the model. Returns (model, history)."""
    model = build_model(X_train.shape[1:], num_classes, hidden_layers=hidden_layers)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "nn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)

    return model, history


def predict_model(model, X_test):
    """ทำการทำนายคลาสจากข้อมูลทดสอบ"""
    probabilities = model.predict(X_test, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)