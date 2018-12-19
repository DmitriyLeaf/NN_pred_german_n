from keras import layers
from keras import Sequential
from keras.utils import np_utils
import time
from keras.callbacks import TensorBoard

MAX_LEN_CODE = 123.0
END_LEN = 10
DROP_OUT = 0.2

X = pickle.load(open("data_sets/x-{}.pickle".format(END_LEN), "rb"))
X = X/MAX_LEN_CODE
#print(X)

Y = pickle.load(open("data_sets/y-{}.pickle".format(END_LEN), "rb"))
Y = np_utils.to_categorical(Y)
#print(Y)

NAME = "German_noun-8dense-ep100-dataAll-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model.add(layers.Dense(END_LEN+20, input_dim=END_LEN, activation='relu'))
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dropout(DROP_OUT))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(X[1000:], Y[1000:], epochs=1000, validation_split=0.25, callbacks=[tensorboard])

model.save('german_noun_dff.model')

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
