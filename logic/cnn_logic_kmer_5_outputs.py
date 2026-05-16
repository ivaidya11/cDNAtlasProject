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

def cnn_kmer_clusters(train, test):

    train_raw = train.copy()
    test_raw = test.copy()

    feature_cols = ['mean_current', 'mean_minus2', 'mean_minus1', 'mean_plus1', 'mean_plus2']
    train_raw[feature_cols] = train_raw[feature_cols].fillna(0)
    test_raw[feature_cols]  = test_raw[feature_cols].fillna(0)

    n_features = len(feature_cols)

    def parse_kmers(df):
        X_list, y_list = [], []
        for trace_id, grp in df.groupby('trace_id'):
            grp = grp.sort_values('step_id')
            if len(grp) != 20:
                continue
            X_list.append(grp[feature_cols].values)
            kmers = grp['cluster_kmer'].apply(
                lambda k: [int(x)-1 if x != 'X' else 0 for x in k.split('_')]
            )
            y_list.append(np.array(kmers.tolist()))  # (20, 5)
        return np.array(X_list), np.array(y_list)    # (n, 20, 5)

    X,      y      = parse_kmers(train_raw)
    X_test, y_test = parse_kmers(test_raw)

    # split into 5 targets — one per k-mer position
    y_L1, y_L2, y_L3, y_L4, y_L5             = [y[:,:,i]      for i in range(5)]
    y_test_L1, y_test_L2, y_test_L3, y_test_L4, y_test_L5 = [y_test[:,:,i] for i in range(5)]

    #  slides a window of size 5 across the 20 timesteps, looking at patterns spanning 5 o nsecutive amino acids, and outputs 64 feature maps
    # Input
    inputs = tf.keras.Input(shape=(20, n_features))

    # Shared layers (same as your current CNN)
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # 5 separate output heads
    out_L1 = tf.keras.layers.Dense(7, activation='softmax', name='L1')(x)
    out_L2 = tf.keras.layers.Dense(7, activation='softmax', name='L2')(x)
    out_L3 = tf.keras.layers.Dense(7, activation='softmax', name='L3')(x)
    out_L4 = tf.keras.layers.Dense(7, activation='softmax', name='L4')(x)
    out_L5 = tf.keras.layers.Dense(7, activation='softmax', name='L5')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[out_L1, out_L2, out_L3, out_L4, out_L5])

    #The Adam optimizer is used for gradient-based optimization. 
    # It adjusts the learning rate based on first and second moments of the gradients. --> picked it by default

    # Categorical cross entropy is used as the loss function for multi-class classification problems.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'L1': 'sparse_categorical_crossentropy',
              'L2': 'sparse_categorical_crossentropy',
              'L3': 'sparse_categorical_crossentropy',
              'L4': 'sparse_categorical_crossentropy',
              'L5': 'sparse_categorical_crossentropy'},
        metrics={'L1': 'accuracy', 'L2': 'accuracy', 'L3': 'accuracy', 'L4': 'accuracy', 'L5': 'accuracy'}
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_L3_accuracy', patience=5, restore_best_weights=True, mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_L3_accuracy', factor=0.5, patience=3, min_lr=1e-5, mode='max')
    ]

    history = model.fit(
        X, {'L1': y_L1, 'L2': y_L2, 'L3': y_L3, 'L4': y_L4, 'L5': y_L5},
        epochs=100, batch_size=32,
        validation_data=(X_test, {'L1': y_test_L1, 'L2': y_test_L2, 'L3': y_test_L3, 'L4': y_test_L4, 'L5': y_test_L5}),
        callbacks=callbacks
    )
    return model, history, X_test, (y_test_L1, y_test_L2, y_test_L3, y_test_L4, y_test_L5)



