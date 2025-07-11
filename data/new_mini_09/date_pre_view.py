import pickle
import json
with open('history_train.pkl',"rb") as file:
    loaded_data=pickle.load(file)
print(loaded_data)
