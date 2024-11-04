#from cbr_fox import cbr_fox
import cbr_fox
from factory.cbr_fox_builder import cbr_fox_builder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import custom_distance

scaler = MinMaxScaler()

df = pd.read_csv("power_usage_2016_to_2020.csv", parse_dates= False)
df

fecha_compuesta = np.array([[row[:4],row[5:7],row[8:10],row[11:13]] for row in df["StartDate"]])

fecha_compuesta = pd.DataFrame(fecha_compuesta,columns=["year","month","day","hour"])

pd.concat([df,fecha_compuesta],axis=1).drop(["StartDate","notes"],axis=1).columns

dataset = pd.concat([df,fecha_compuesta],axis=1).drop(["StartDate","notes"],axis=1)
dataset = scaler.fit_transform(dataset)
dataset

def split_sequences(sequences, n_steps):
    inputnn, target = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix + 1 > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix, 0]
        inputnn.append(seq_x)
        target.append(seq_y)
    return np.array(inputnn), np.array(target)

inputnn,target_training_windows = split_sequences(dataset,7)

input_train, input_test, target_train, target_test = train_test_split(inputnn, target_training_windows, test_size = 0.30, random_state=4,shuffle=True)


nonOutputColumns=[1,2,3,4,5]
training_windows = np.delete(inputnn, nonOutputColumns, 2)
forecasted_window, training_windows = training_windows[-1], training_windows[:-1]
target_training_windows = target_training_windows[:-1]
windowsLen = len(training_windows)
componentsLen = training_windows.shape[2]
windowLen = training_windows.shape[1]
prediction = [0.09680558]

import numpy as np

# Assuming all variables are defined as in your code
np.savez("variables.npz",
         training_windows=training_windows,
         forecasted_window=forecasted_window,
         target_training_windows=target_training_windows,
         windowsLen=windowsLen,
         componentsLen=componentsLen,
         windowLen=windowLen,
         prediction=prediction)

import numpy as np

# Load the saved data
data = np.load("variables.npz")

# Retrieve each variable
training_windows = data['training_windows']
forecasted_window = data['forecasted_window']
target_training_windows = data['target_training_windows']
windowsLen = data['windowsLen'].item()  # Extract single value from array
componentsLen = data['componentsLen'].item()
windowLen = data['windowLen'].item()
prediction = data['prediction']


from custom_distance.cci_distance import cci_distance
techniques = [("CCI", {"punishedSumFactor":.5}), ("CCI", {"punishedSumFactor":.7})]
techniques = [
    cbr_fox.cbr_fox(metric=cci_distance,kwargs={"punishedSumFactor":.5})
]
p = cbr_fox_builder(techniques)
p.fit(training_windows = training_windows,target_training_windows = target_training_windows.reshape(-1,1), forecasted_window = forecasted_window)
p.predict(prediction = prediction,num_cases=5)
p.plot_correlation()
