import pandas as pd
from sklearn.tree import DecisionTreeClassifier
music_data = pd.read_csv('music.csv')
input_data = music_data.drop(columns=['genre'])
output_data = music_data['genre']

model = DecisionTreeClassifier()
model.fit(input_data, output_data)
predictions = model.predict([ [21, 1] ])
# print(input_data)
# print(output_data)
print(predictions)