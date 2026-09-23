import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, conv_filters=(32, 64, 128),
                dense_units=128, norm_stats=None):
    """CNN: one conv block per entry in conv_filters -> dense head.

    conv_filters : filters of each Conv2D block, e.g. (32, 64) = 2 conv layers
    dense_units  : neurons in the fully connected layer
    norm_stats   : {"mean": [...], "std": [...]} from the training set
    """

    stack = [keras.Input(shape=input_shape)]

    # Standardization (x - mean) / std inside the model, using TRAIN statistics.
    # Being part of the model means inference code never has to remember it.
    if norm_stats is not None:
        stack.append(layers.Normalization(
            axis=-1,
            mean=norm_stats["mean"],
            variance=[s ** 2 for s in norm_stats["std"]],
        ))
    else:
        stack.append(layers.Rescaling(1.0 / 255))

    # Light augmentation, only active during training.
    # This is what keeps a small CNN from memorising the training set.
    stack += [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]

    for filters in conv_filters:
        stack += [
            layers.Conv2D(filters, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
        ]

    stack += [
        # Pooling instead of Flatten keeps the dense head small
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(0.3),

        # 1 sigmoid output for 2 classes, softmax otherwise
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ]

    model = keras.Sequential(stack)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class EpochSnapshot(keras.callbacks.Callback):
    """Record train/val/test metrics after chosen epochs (epoch comparison).

    One long run is enough: the first N epochs of a 30-epoch run are exactly
    an N-epoch run, because nothing in the schedule depends on the total.
    """

    def __init__(self, config_name, checkpoints, X_test, y_test):
        super().__init__()
        self.config_name = config_name
        self.checkpoints = set(checkpoints)
        self.X_test = X_test
        self.y_test = y_test
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        n = epoch + 1
        if n not in self.checkpoints:
            return

        _, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.rows.append({
            "config": self.config_name,
            "epochs": n,
            "train_acc": float(logs["accuracy"]),
            "train_loss": float(logs["loss"]),
            "val_acc": float(logs["val_accuracy"]),
            "val_loss": float(logs["val_loss"]),
            "test_acc": float(test_acc),
        })


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32,
                conv_filters=(32, 64, 128), dense_units=128,
                norm_stats=None, extra_callbacks=(), model_name="cnn_model"):
    """Build, train and save the model. Returns (best_model, history).

    No EarlyStopping here: every configuration must train for the same number
    of epochs, otherwise the epoch comparison is not comparable. The weights
    of the epoch with the best VALIDATION accuracy are kept instead.
    """

    model = build_model(X_train.shape[1:], num_classes,
                        conv_filters, dense_units, norm_stats)
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        *extra_callbacks,
    ]

    ckpt_path = None
    if output_dir:
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        ckpt_path = os.path.join(models_dir, f"{model_name}.keras")
        callbacks.append(keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True
        ))

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        with open(os.path.join(output_dir, f"history_{model_name}.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        # Continue with the best-validation weights, not the last epoch
        model = keras.models.load_model(ckpt_path)
        print(f"Saved: {ckpt_path}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)
