import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

test_data = sys.argv[1]
model_file = sys.argv[2]
sys_output = sys.argv[3]
test_formatted = open(os.path.join(os.path.dirname(__file__), test_data), 'r').read().split("\n")[:-1]
model_formatted = open(os.path.join(os.path.dirname(__file__), model_file), 'r').read().split("\n")[:-1]

# get relevant information out of the model file
svm_type = model_formatted[0].split()[-1]
kernel_type = model_formatted[1].split()[-1]
sv = 0
for i in range(len(model_formatted)):
    if model_formatted[i] == "SV":
        sv = i
        break

total_sv = int(model_formatted[sv - 4].split()[-1])
rho = float(model_formatted[sv - 3].split()[-1])
labels = model_formatted[sv - 2].split()[1:]
c_0 = labels[0]
c_1 = labels[1]
nr_sv = model_formatted[sv - 1].split()[1:]
params = model_formatted[2:sv - 4]

model_vectors = model_formatted[sv + 1:]

# set default values, in case we don't run into them in the file
degree = None
gamma = None
coef0 = None

# set parameters
for parameter in params:
    if parameter.startswith("d"):
        degree = int(parameter.split()[-1])
    elif parameter.startswith("g"):
        gamma = float(parameter.split()[-1])
    elif parameter.startswith("c"):
        coef0 = float(parameter.split()[-1])
    else:
        pass


# find minimum vector dimensionality
def find_highest(dataset):
    highest = 0
    for line in dataset:
        entry = line.split()
        for f_v in entry[1:]:
            f = int(f_v.split(":")[0])
            if f > highest:
                highest = f
    return highest


model_high = find_highest(model_vectors)
data_high = find_highest(test_formatted)
if model_high > data_high:
    overall_high = model_high
else:
    overall_high = data_high

# create an array to represent the model and store the weights as well
model = np.zeros((total_sv, overall_high+1))
weights = []
for i in range(total_sv):
    sv = model_vectors[i].split()
    weight = float(sv[0])
    weights.append(weight)
    for f_v in sv[1:]:
        f_v = f_v.split(":")
        f = int(f_v[0])
        v = int(f_v[1])
        model[i, f] = v

# determine which function to use
if kernel_type == "linear":
    k = lambda u, y: np.dot(u, y)
elif kernel_type == "polynomial":
    k = lambda u, y: ((gamma * np.dot(u, y)) + coef0) ** degree
elif kernel_type == "rbf":
    k = lambda u, y: np.exp(-1 * gamma * np.dot((u - y), (u - y)))
else:
    k = lambda u, y: np.tanh(gamma * np.dot(u, y) + coef0)


# takes a test instance and a kernel function and returns a predicted class based on the model given
def classify(test_inst, k):
    output = 0
    for i in range(len(model)):
        output += (weights[i] * k(model[i], test_inst))
    f_x = output - rho
    if f_x >= 0:
        return 0, f_x
    else:
        return 1, f_x


y_true = []
y_pred = []
# write results to file
with open(sys_output, 'w') as s:
    for i in range(len(test_formatted)):
        instance = test_formatted[i].split()
        actual = int(instance[0])
        test_vec = np.zeros(overall_high+1)
        for f_v in instance[1:]:
            f_v = f_v.split(":")
            f = int(f_v[0])
            v = int(f_v[1])
            test_vec[f] = v
        predicted, f_x = classify(test_vec, k)
        y_pred.append(predicted)
        y_true.append(actual)
        s.write(str(actual) + " " + str(predicted) + " " + "%.5f" % f_x + "\n")

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)