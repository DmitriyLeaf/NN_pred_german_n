import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# It reads dataset and made usable format with diffrent words lens

MAX_LEN_CODE = 123.0
END_LEN = [8, 10, 15, 20]
F_NAME = "data_sets/dict.cc_nouns_with_gender.txt"
NUM_DATA = 10

def readFileRec(file_name, len_str):
	training_data_file = open(file_name, 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	training_data_list = training_data_list[1:]
	print(training_data_list[:10])
	np.random.shuffle(training_data_list)
	training_data_list = training_data_list[:NUM_DATA]
	print(training_data_list[:10])

	f_inputs = [[[0] for i in range(0, len_str)]]
	targets = []
	f_inputs = np.array(f_inputs)

	for record in training_data_list:
		all_values = record.split(' ')
		all_values = [word.replace("\n", "") for word in all_values]
		i = [0]
		for letter in all_values[1]:
			i = np.vstack((i, np.array([ord(letter.lower())])))
			pass
		inputs = i
		if(len(inputs) > len_str):
			inputs = inputs[len(inputs)-len_str:]
		elif(len(inputs) < len_str):
			begin = np.zeros((len_str - len(inputs)), dtype=np.float)
			for b in begin:
			 	inputs = np.vstack((b, inputs))
			 	pass
		f_inputs = np.vstack((f_inputs, [inputs]))
		targets.append(all_values[0])
	f_inputs = np.delete(f_inputs, 0, 0)
	encoder = LabelEncoder()
	encoder.fit(targets)
	new_targets = encoder.transform(targets)
	f_targets = new_targets
	return f_inputs, f_targets

for lens in END_LEN:
	X, Y = readFileRec(F_NAME, lens)

	pickle_out = open("x-{}.pickle".format(lens), "wb")
	pickle.dump(X, pickle_out)
	pickle_out.close()

	pickle_out = open("y-{}.pickle".format(lens), "wb")
	pickle.dump(Y, pickle_out)
	pickle_out.close()

#pickle_in = open("x-8.pickle", "rb")
#x = pickle.load(pickle_in)
#print(x)

#pickle_in = open("y-8.pickle", "rb")
#y = pickle.load(pickle_in)
#print(y)