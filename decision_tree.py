import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
input_data = music_data.drop(columns=['genre'])
output_data = music_data['genre']

model = DecisionTreeClassifier()
model.fit(input_data, output_data)

tree.export_graphviz(model, out_file='music_dec_tree.dot',
                     feature_names=['age','gender'], class_names=sorted(output_data.unique()),
                     label='all',filled=True, rounded=True)