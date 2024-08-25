from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
import pandas as pd


# dataset https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
data_raw = datasets.load_breast_cancer()
data = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
data["target"] = data_raw.target

train, test = tts(data, test_size=0.3)