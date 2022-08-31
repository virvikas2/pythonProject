import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals import joblib
import joblib
music_data = pd.read_csv('music.csv')
input_data = music_data.drop(columns=['genre'])
output_data = music_data['genre']

model = DecisionTreeClassifier()
model.fit(input_data.values, output_data)
joblib.dump(model, 'music_persist_model.joblib')
# predictions = model.predict([ [21, 1], [22, 0]])
# print(predictions)