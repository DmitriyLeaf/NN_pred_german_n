import tensorflow as tf
from keras import layers
from keras import Sequential
from keras.utils import np_utils
import time
import pickle
from keras.callbacks import TensorBoard

MAX_LEN_CODE = 123.0
END_LEN = 20
F_NAME = "data_sets/dict.cc_nouns_with_gender.txt"
DROP_OUT = 0.2

X = pickle.load(open("x-{}.pickle".format(END_LEN), "rb"))
X = X/MAX_LEN_CODE
print(X[:5])

Y = pickle.load(open("y-{}.pickle".format(END_LEN), "rb"))
Y = np_utils.to_categorical(Y)
print(Y[:5])

NAME = "German_noun-4Lstm(128,200,300,128)2dense(60,3)-ep100-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


model = Sequential()

model.add(layers.LSTM(128, input_shape=(X.shape[1:]), activation='relu', return_sequences=True))
model.add(layers.Dropout(DROP_OUT))
model.add(layers.LSTM(200, activation='relu', return_sequences=True))
model.add(layers.LSTM(300, activation='relu', return_sequences=True))
model.add(layers.LSTM(128, activation='relu'))
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(X[1000:], Y[1000:], epochs=100, validation_split=0.25, callbacks=[tensorboard])

prediction = model.predict_classes(X[:1000])
answer = [list(i).index(1) for i in Y[:1000]]
score = 0
for i in range(len(answer)):
    if answer[i] == prediction[i]:
        score += 1
        pass
    pass
print(score)
print(len(answer))
result = 100 * score/len(answer)
print("{}%".format(round(result)))

scores = model.evaluate(X[:1000], Y[:1000], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(scores)

print("\n")