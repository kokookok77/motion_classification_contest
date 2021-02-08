import pandas as pd
import numpy as np

import tensorflow as tf



train_feature = pd.read_csv('sampled_feature_2.csv')
train_label = pd.read_csv('sampled_label_2.csv')

y=tf.keras.utils.to_categorical(train_label['label']) 
print(y.shape)

X=tf.reshape(np.array(train_feature.iloc[:,3:9]),[-1, 600, 6])
print(X.shape)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout


DATASET_INDEX = 3

MAX_TIMESTEPS = 600
MAX_NB_VARIABLES = 6
NB_CLASS = 61

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_TIMESTEPS, MAX_NB_VARIABLES))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
   
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # add load model code here to fine-tune

    return model


model = generate_model()
model.fit(X,y, epochs=30, batch_size=128, validation_split=0.2)


test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

test_X=tf.reshape(np.array(test.iloc[:,2:]),[-1, 600, 6])
print(test_X.shape)

prediction=model.predict(test_X)

submission.iloc[:,1:]=prediction
submission.to_csv('first_submission.csv', index=False)