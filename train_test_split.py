import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
input_data = music_data.drop(columns=['genre'])
output_data = music_data['genre']
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(input_train, output_train)
predictions = model.predict(input_test)
score = accuracy_score(output_test, predictions)
print(score)
# print(predictions)