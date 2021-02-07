import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D




train_feature = pd.read_csv('sampled_feature_2.csv')
train_label = pd.read_csv('sampled_label_2.csv')

y=tf.keras.utils.to_categorical(train_label['label']) 
print(y.shape)

X=tf.reshape(np.array(train_feature.iloc[:,3:9]),[-1, 600, 6])
print(X.shape)


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, num_filters=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.activation = tf.keras.activations.get(activation)

    def _default_Conv1D(self, filters, kernel_size):
        return tf.keras.layers.Conv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding='same',
                                      activation='relu',
                                      use_bias=False)

    def call(self, inputs):
        
        z_bottleneck = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(inputs)
        z_maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(inputs)

        z1 = self._default_Conv1D(filters=self.num_filters, kernel_size=10)(z_bottleneck)
        z2 = self._default_Conv1D(filters=self.num_filters, kernel_size=20)(z_bottleneck)
        z3 = self._default_Conv1D(filters=self.num_filters, kernel_size=40)(z_bottleneck)
        z4 = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(z_maxpool)

        z = tf.keras.layers.Concatenate(axis=2)([z1, z2, z3, z4])
        z = tf.keras.layers.BatchNormalization()(z)

        return self.activation(z)


def shortcut_layer(inputs, z_inception):
    z_shortcut = tf.keras.layers.Conv1D(filters=int(z_inception.shape[-1]), 
                                        kernel_size=1, 
                                        padding='same', 
                                        use_bias=False)(inputs)

    z_shortcut = tf.keras.layers.BatchNormalization()(z_shortcut)

    z = tf.keras.layers.Add()([z_shortcut, z_inception])

    return tf.keras.layers.Activation('relu')(z)


def build_model(input_shape, num_classes, num_modules=6):
    input_layer = tf.keras.layers.Input(input_shape)
    z = input_layer
    z_residual = input_layer

    for i in range(num_modules):
        z = InceptionModule()(z)
        if i%3 == 2:
            z = shortcut_layer(z_residual, z)
            z_residual = z

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(z)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = tf.keras.models.Model(inputs= input_layer, outputs= output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


model = build_model((600, 6), 61, num_modules=6)
model.fit(X,y, epochs=30, batch_size=128, validation_split=0.2)


test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

test_X=tf.reshape(np.array(test.iloc[:,2:]),[-1, 600, 6])
print(test_X.shape)

prediction=model.predict(test_X)

submission.iloc[:,1:]=prediction
submission.to_csv('first_submission.csv', index=False)