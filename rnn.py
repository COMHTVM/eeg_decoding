
import numpy as np
import models
import utils
import sys
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

numTests = 50 
batch_size = 64
num_classes = 4
num_epochs = 100
model_name = "cnn"
model_dir_name = "models"   

# Get data
personId = list(np.arange(9))
X_train, X_test, y_train, y_test = utils.get_down2_2d_data(personId, numTests)

# print(X_train.shape)
# for i in np.arange(22):
    # scaler = preprocessing.StandardScaler().fit(X_train[:, :, i])
    # X_train[:, :, i] = scaler.transform(X_train[:, :, i])
    # X_test[:, :, i] = scaler.transform(X_test[:, :, i])

# Categorize labels
y_train, y_test = utils.categorize_labels(y_train, y_test, num_classes)

# Get current model
model = models.get_rnn_model()

# Train model
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.2, callbacks=[earlystop])
 
# Print testing error
print("##### Overall testing results")
print(model.evaluate(X_test, y_test))
for i in np.arange(9):
    numData = X_test.shape[0]//9
    X_sub_te = X_test[i*numData:(i+1)*numData]
    y_sub_te = y_test[i*numData:(i+1)*numData]
    print("###### Evaluation results for subject ", i+1)
    print(model.evaluate(X_sub_te, y_sub_te))
