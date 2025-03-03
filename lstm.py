#%% load dataset and preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder # numeralize the data
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # array padding
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

newsgroup = fetch_20newsgroups(subset = "all", download_if_missing=True) # gets the whole dataset
x = newsgroup.data
y = newsgroup.target
 
# tokenize the dataset
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
X_sequences = tokenizer.texts_to_sequences(x)
X_padded = pad_sequences(X_sequences, maxlen=100, padding="post", truncating="post")

# label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#train test split
x_train, x_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size = 0.2, random_state=42)

#%% build the lstm model
from tensorflow.keras import backend as K




def f1_score(y_true, y_pred):
    num_classes = 20  # Set this to the number of classes in your problem
    
    # Convert y_true (sparse labels) to one-hot encoded tensor of shape (batch_size, num_classes)
    y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes)
    y_true = K.cast(y_true, 'float32')
    
    # Round y_pred to get binary predictions
    y_pred = K.round(y_pred)
    
    tp = K.sum(y_pred * y_true, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)



def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation="softmax"))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", f1_score])
    return model  

# model olusturma
model = build_lstm_model()
model.build(input_shape=(None, 100))  
model.summary()

#%% model training

# callback: early stop
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
# model training
history = model.fit(x_train, y_train,
                    epochs = 30,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stopping])


#%% model evaluation

# evaluation via test datasets
loss, accuracy, f1 = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")

# visualization of accuracy and loss using history
plt.figure()

#training loss and val loss

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")
plt.show()
         
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")
plt.show()




