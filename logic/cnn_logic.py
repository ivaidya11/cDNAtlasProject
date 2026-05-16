import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from logic.database import AA_PROPERTIES
import pandas as pd
from logic.clustering_cnn import get_clusters
# train_raw = pd.read_csv('data/nanopore_feature_by_cluster.csv')
from sklearn.utils.class_weight import compute_class_weight
# Cnn works by sliding a small filter ove rthe input and detecting local pattersn, each filter learns to recognize a specific feature, such as patterns acorss considectuive 
# amino acids in a trace
# instead of connecting every input to every neron, convolution preserves the structur eof hte input sequences and theweights across positions

#

def cnn_sweep(n_clusters, train, test):

    train_raw = train.copy()
    test_raw = test.copy()

    aa_to_cluster = get_clusters(n_clusters)

    train_raw['label'] = train_raw['amino_acid'].map(aa_to_cluster) - 1


    feature_cols = ['mean_current', 'mean_minus2', 'mean_minus1', 'mean_plus1', 'mean_plus2']
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
        #  slides a window of size 5 across the 20 timesteps, looking at patterns spanning 5 o nsecutive amino acids, and outputs 64 feature maps
        tf.keras.layers.Conv1D(64,  kernel_size=5, padding='same', activation='relu', input_shape=(20, n_features)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # deeper layer that combines the 64 low-level patterns into 128 higher level patterns, also looking at spans of 5 amino acids
        tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # finer local patterns, only size 3 for finer grained refinement, without looking at too wide a context, can also make it 5 wont really affect anything i think
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        # CNN + BiLSTM (adds full-sequence context):

        # after the cnn extracts features the bilstm can add global sequence context, itreads the trace forward and backward
        # so that each positions prediction is informed by the entire trace, not jsut he neighborhod
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),

        # squishes the bilstm output into a compct representation
        tf.keras.layers.Dense(64, activation='relu'),

        # outputs a probability distribution over the n_clusters classes for each of the 20 positions,
        #  and you predict the class with the highest probability as the cluster label for each amino acid in the trace

        tf.keras.layers.Dense(n_clusters, activation='softmax')
    ])
    #The Adam optimizer is used for gradient-based optimization. 
    # It adjusts the learning rate based on first and second moments of the gradients. --> picked it by default

    # Categorical cross entropy is used as the loss function for multi-class classification problems.
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
    # classes = np.unique(y)
    # weights = compute_class_weight(class_weight='balanced', classes=classes, y=y.flatten())
    # # sample_weight must be (n_samples, 20) for sequence output; map each timestep label to its weight
    # weight_map = dict(zip(classes, weights))
    # sample_weight = np.vectorize(weight_map.get)(y).astype(np.float32)  # (n_traces, 20)

    history = model.fit(X, y, epochs=100, batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks)
    return model, history, X_test, y_test



