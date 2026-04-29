import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from logic.database import AA_PROPERTIES
import pandas as pd
from logic.clustering_cnn import get_clusters
# train_raw = pd.read_csv('data/nanopore_feature_by_cluster.csv')


def cnn_sweep(n_clusters, train, test):

    train_raw = train.copy()
    test_raw = test.copy()

    aa_to_cluster = get_clusters(n_clusters)

    train_raw['label'] = train_raw['amino_acid'].map(aa_to_cluster) - 1


    feature_cols = ['mean_current', 'weighted_window_current', 'mean_minus2', 'mean_minus1', 'mean_plus1', 'mean_plus2']
    train_raw[feature_cols] = train_raw[feature_cols].fillna(0)

    n_features = len(feature_cols)

    X_list, y_list = [], []
    for trace_id, grp in train_raw.groupby('trace_id'):
        grp = grp.sort_values('step_id')
        if len(grp) != 20:
            continue
        X_list.append(grp[feature_cols].values)
        y_list.append(grp['label'].values)

    X = np.array(X_list)   # (n_traces, 20, n_features)
    y = np.array(y_list)   # (n_traces, 20)

    test_raw['label'] = test_raw['amino_acid'].map(aa_to_cluster) - 1
    test_raw[feature_cols] = test_raw[feature_cols].fillna(0)

    X_test_list, y_test_list = [], []
    for trace_id, grp in test_raw.groupby('trace_id'):
        grp = grp.sort_values('step_id')
        if len(grp) != 20:
            continue
        X_test_list.append(grp[feature_cols].values)
        y_test_list.append(grp['label'].values)

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64,  kernel_size=5, padding='same', activation='relu', input_shape=(20, n_features)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_clusters, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5)
    ]

    history = model.fit(X, y, epochs=100, batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks)
    return model, history, X_test, y_test



