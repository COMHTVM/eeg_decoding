
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv1D, AveragePooling2D, GRU, LSTM, AveragePooling1D, MaxPooling1D, TimeDistributed
from keras.layers.core import Reshape

num_classes = 4

def get_model1():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(25, kernel_size=(10, 1), input_shape=(1000, 22, 1)))
    model.add(Conv2D(25, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((991, 25, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 2
    model.add(Conv2D(50, kernel_size=(10, 25)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((321, 50, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 3
    model.add(Conv2D(100, kernel_size=(10, 50)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((98, 100, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 4
    model.add(Conv2D(200, kernel_size=(10, 100)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((23, 200, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 5
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
def get_model2():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(25, kernel_size=(10, 1), input_shape=(1000, 22, 1)))
    model.add(Conv2D(25, kernel_size=(1, 22)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Reshape((991, 25, 1)))
    model.add(MaxPooling2D(pool_size=(5, 1)))

    ## Layer 2
    model.add(Conv2D(50, kernel_size=(10, 25)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Reshape((189, 50, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 3
    model.add(Conv2D(100, kernel_size=(10, 50)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Reshape((54, 100, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 4
    model.add(Conv2D(200, kernel_size=(10, 100)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Reshape((9, 200, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 5
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
def get_cropped_model1():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(25, kernel_size=(10, 1), input_shape=(500, 22, 1)))
    model.add(Conv2D(25, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((491, 25, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 2
    model.add(Conv2D(50, kernel_size=(10, 25)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((154, 50, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 3
    model.add(Conv2D(100, kernel_size=(10, 50)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((42, 100, 1)))
    model.add(MaxPooling2D(pool_size=(3, 1)))

    ## Layer 4
    model.add(Conv2D(200, kernel_size=(10, 100)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((5, 200, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    ## Layer 5
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
def get_model3():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(40, kernel_size=(25, 1), input_shape=(1000, 22, 1)))
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((976, 40, 1)))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1)))

    ## Layer 2
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
def get_model4():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(40, kernel_size=(25, 1), input_shape=(1000, 22, 1)))
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((976, 40, 1)))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1)))
    
    
    model.add(Conv2D(80, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    # model.add(Reshape((976, 40, 1)))
    model.add(AveragePooling2D(pool_size=(5, 5)))

    ## Layer 2
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
def get_model5():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(40, kernel_size=(25, 1), input_shape=(1000, 22, 1)))
    # model.add(Activation('elu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Reshape((976, 40, 1)))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1)))

    ## Layer 2
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
def get_spec_model():
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(24, kernel_size=(2, 12), input_shape=(8, 257, 22), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Layer 2
    model.add(Conv2D(48, kernel_size=(2, 8), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Layer 3
    model.add(Conv2D(96, kernel_size=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
def get_cropped_model2():
    model = Sequential()

    ## Layer 1
    model.add(Conv2D(40, kernel_size=(25, 1), input_shape=(500, 22, 1)))
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((476, 40, 1)))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1)))

    ## Layer 2
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
    
def get_feat_model():
    model = Sequential()

    ## Layer 1
    model.add(Reshape((513, 22, 1), input_shape=(513, 22)))
    model.add(Conv2D(40, kernel_size=(10, 1)))
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Reshape((504, 40, 1)))
    model.add(AveragePooling2D(pool_size=(15, 1), strides=(15, 1)))

    ## Layer 2
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
def get_rnn_model():
    model = Sequential()

    ## Layer 1
    # model.add(AveragePooling1D(pool_size=2, input_shape=(1000, 22)))
    model.add(GRU(128, dropout=0.05, input_shape=(500, 22), return_sequences=True))
    model.add(GRU(128, dropout=0.05, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
def get_rnn_model2():
    model = Sequential()

    ## Layer 1
    # model.add(AveragePooling1D(pool_size=2, input_shape=(1000, 22)))
    model.add(Conv2D(40, kernel_size=(25, 1), input_shape=(1000, 22, 1)))
    # model.add(Activation('elu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    
    model.add(Conv2D(40, kernel_size=(1, 22)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Reshape((976, 40, 1)))
    model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1)))
    model.add(Reshape((61, 40)))
    
    model.add(GRU(128, dropout=0.05, return_sequences=True))
    model.add(GRU(128, dropout=0.05, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
    
def get_rnn_model3():
    model = Sequential()

    ## Layer 1
    
    model.add(SimpleRNN(32, activation='sigmoid', return_sequences=True, input_shape=(1000, 22)))
    model.add(TimeDistributed(Dense(output_dim=num_classes)))
    model.add(Activation("softmax"))
    
    
    # model.add(LSTM(64, dropout=0.5, activation='tanh', return_sequences=True))
    # model.add(Conv1D(1, kernel_size=25, strides=5, input_shape=(1000, 22)))
    # model.add(Reshape((-1, 22)))
    # model.add(Reshape((1000, 22, 1), input_shape=(1000, 22)))
    # model.add(Conv2D(40, kernel_size=(25, 1)))
    # model.add(Conv2D(40, kernel_size=(1, 22)))
    # model.add(Reshape((976, 40)))
    # model.add(BatchNormalization())
    # model.add(Activation('elu'))
    # model.add(Dropout(0.5))
    # model.add(Reshape((496, 22)))
    # model.add(MaxPooling2D(pool_size=(111115, 1), strides=(15, 1)))
    # model.add(Reshape((65, 40)))

    # model.add(Conv1D(40, kernel_size=10, strides=2))
    # model.add(Conv1D(40, kernel_size=10, strides=2))
    # model.add(GRU(32, dropout=0.2, activation='tanh', return_sequences=True))
    # model.add(GRU(64, dropout=0.2, activation='tanh', return_sequences=True))
    # model.add(GRU(32, dropout=0.2, activation='tanh'))
    # model.add(LSTM(64, dropout=0.2, activation='tanh', return_sequences=True))
    # model.add(LSTM(64, dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
def get_rnn_model4():
    model = Sequential()

    ## Layer 1
    model.add(GRU(128, dropout=0.05, input_shape=(8, 257, 22), return_sequences=True))
    # model.add(GRU(128, dropout=0.05, return_sequences=True))
    model.add(GRU(128, dropout=0.05))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
    
def get_dnn_model():
    model = Sequential()
    
    model.add(Dense(256, input_shape=(24*22, )))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model