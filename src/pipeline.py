import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def basic_features(df):
    df = df.copy()
    df['Is_Male'] = (df['Sex'] == 'male').astype(int)
    df['Deck'] = df['Cabin'].str[0]
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Section'] = np.select(
        [(df['Pclass'] == 3) & (df['Parch'] == 0) & (df['Sex'] == 'male'),
         df['Pclass'] == 3,
         df['Pclass'] == 1],
        ['bow', 'stern', 'midship'],
        default = 'midship_aft'
    )
    return df.drop(columns = ['Sex','Cabin','SibSp','Parch'])


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group = 'Pclass', cols = ('Age','Fare')):
        self.group = group
        self.cols = cols

    def fit(self, X, y=None):
        self.medians_ = {c: X.groupby(self.group)[c].median().to_dict() for c in self.cols}
        self.global_ = {c: X[c].median() for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].fillna(X[self.group].map(self.medians_[c])).fillna(self.global_[c])
        return X
    

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ticket_counts_ = X['Ticket'].value_counts().to_dict()
        fare_per_person = X['Fare'] / X['Ticket'].map(self.ticket_counts_).fillna(1)
        Q1, Q3 = fare_per_person.quantile([0.25, 0.75])
        self.upper_bound_ = round(Q3 + 1.5 * (Q3 - Q1), 2)
        return self
    def transform(self, X):
        X = X.copy()
        people_on_ticket = X['Ticket'].map(self.ticket_counts_).fillna(1)
        fare = X['Fare'] / people_on_ticket
        X['Fare_per_person'] = fare
        X['Wealth_level'] = np.select(
            [fare >= 200, fare >= self.upper_bound_, fare >= 30, fare >= 12],
            ['elite', 'wealthy', 'upper_class', 'middle_class'],
            default= 'lower_class'
        )
        return X.drop(columns=['Fare', 'Ticket'])
    

wealth_categories = [['elite', 'wealthy', 'upper_class', 'middle_class', 'lower_class']]

columns = ColumnTransformer(
    [
        ('section', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Section']),
        ('embarked', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown= 'ignore', sparse_output=False)),
        ]), ['Embarked']),
        ('deck', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='U')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)),
        ]), ['Deck']),
        ('wealth', OrdinalEncoder(categories = wealth_categories), ['Wealth_level']),
        ('age', MinMaxScaler(), ['Age']),
        ('fare', FunctionTransformer(np.log1p), ['Fare_per_person']),
        ('rest', 'passthrough', ['Pclass', 'Is_Male', 'Family_size'])    
    ],
    remainder='drop'
)


pipeline = Pipeline([
    ('basic', FunctionTransformer(basic_features)),
    ('impute', GroupMedianImputer()),
    ('features', FeatureEngineering()),
    ('cols', columns)
])
   
   