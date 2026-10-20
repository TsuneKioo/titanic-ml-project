import pipeline
import configs
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/raw/titanic.csv", index_col= 'PassengerId')

X = df.drop(columns = ['Survived', 'Name'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle = True, stratify=y, random_state = configs["data"]["seed"])

pipeline.pipeline.named_steps["cols"].set_output(transform="pandas")

X_train = pipeline.pipeline.fit_transform(X_train)
X_test = pipeline.pipeline.transform(X_test)

X_train.to_parquet('../data/processed/train_data.parquet')
X_test.to_parquet('../data/processed/test_data.parquet')

y_train.to_frame().to_parquet('../data/processed/y_train.parquet')
y_test.to_frame().to_parquet('../data/processed/y_test.parquet')