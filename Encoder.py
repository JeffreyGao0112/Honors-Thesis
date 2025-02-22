import numpy as np
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler

def create_autoencoder(input_dim, encoding_dim):
    """Creates an autoencoder model."""
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='linear')(encoder)  # or 'linear'

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)  # Separate encoder

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder_model

