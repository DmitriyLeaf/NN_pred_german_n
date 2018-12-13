import tensorflow as tf
from keras import layers
from keras import Sequential
import numpy as np
import codecs
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time
from keras.callbacks import TensorBoard

MAX_LEN_CODE = 123.0
END_LEN = 20
F_NAME = "data_sets/dict.cc_nouns_with_gender.txt"
DROP_OUT = 0.2

def readFile(file_name, MAX_LEN_CODE, END_LEN):
	training_data_file = open(file_name, 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	training_data_list = training_data_list[1:]
	print(training_data_list[:10])
	np.random.shuffle(training_data_list)
	print(training_data_list[:10])

	f_inputs = [[[0] for i in range(0, END_LEN)]]
	targets = []
	f_inputs = np.asfarray(f_inputs)

	for record in training_data_list:
		all_values = record.split(' ')
		all_values = [word.replace("\n", "") for word in all_values]
		i = [0]
		for letter in all_values[1]:
			i = np.vstack((i, np.asfarray([ord(letter.lower())/MAX_LEN_CODE])))
			pass
		inputs = i
		if(len(inputs) > END_LEN):
			inputs = inputs[len(inputs)-END_LEN:]
		elif(len(inputs) < END_LEN):
			begin = np.zeros((END_LEN - len(inputs)), dtype=np.float)
			for b in begin:
			 	inputs = np.vstack((b, inputs))
			 	pass
		f_inputs = np.vstack((f_inputs, [inputs]))

		targets.append(all_values[0])

	f_inputs = np.delete(f_inputs, 0, 0)
	encoder = LabelEncoder()
	encoder.fit(targets)
	new_targets = encoder.transform(targets)
	
	f_targets = np_utils.to_categorical(new_targets)
	print("Read f done!")
	return f_inputs, f_targets

X, Y = readFile(F_NAME, MAX_LEN_CODE, END_LEN)

NAME = "German_noun-drop-{}-len-{}-{}".format(DROP_OUT, END_LEN, int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model.add(layers.LSTM(60, input_shape=(X.shape[1:]), activation='relu', return_sequences=True))
model.add(layers.Dropout(DROP_OUT))
model.add(layers.LSTM(128, activation='relu'))
model.add(layers.Dense(60, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer="adam",
			  loss="categorical_crossentropy",
			  metrics=['accuracy'])

model.fit(X[1000:], Y[1000:], epochs=100, validation_split=0.25, callbacks=[tensorboard])

prediction = model.predict_classes(X[:1000])
np.set_printoptions(threshold=np.nan)
#print(prediction)
answer = [list(i).index(1) for i in Y[:1000]]
score = 0
for i in range(answer):
	if answer[i] == prediction[i]:
		score += 1
print(score)
print(len(answer))
result = 100 * score/len(answer)
print("{}%".format(round(result)))

scores = model.evaluate(X[:1000], Y[:1000], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(scores)

print("\n")
