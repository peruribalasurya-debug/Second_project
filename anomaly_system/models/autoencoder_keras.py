from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KerasAEConfig:
    hidden_sizes: list[int]
    bottleneck: int
    dropout: float
    l2: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int


def build_autoencoder(input_dim: int, cfg: KerasAEConfig):
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(cfg.l2) if cfg.l2 and cfg.l2 > 0 else None
    inp = tf.keras.Input(shape=(input_dim,), name="x")
    x = inp
    for i, h in enumerate(cfg.hidden_sizes):
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg, name=f"enc_dense_{i}")(x)
        if cfg.dropout and cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout, name=f"enc_dropout_{i}")(x)
    z = tf.keras.layers.Dense(cfg.bottleneck, activation="relu", kernel_regularizer=reg, name="bottleneck")(x)

    y = z
    for i, h in enumerate(reversed(cfg.hidden_sizes)):
        y = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg, name=f"dec_dense_{i}")(y)
        if cfg.dropout and cfg.dropout > 0:
            y = tf.keras.layers.Dropout(cfg.dropout, name=f"dec_dropout_{i}")(y)
    out = tf.keras.layers.Dense(input_dim, activation="linear", name="recon")(y)

    model = tf.keras.Model(inp, out, name="autoencoder")
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="mse")
    return model


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: KerasAEConfig,
):
    import tensorflow as tf

    model = build_autoencoder(X_train.shape[1], cfg)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    return model, history


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    recon = model.predict(X, verbose=0)
    err = np.mean((X - recon) ** 2, axis=1)
    return err.astype(np.float64)

