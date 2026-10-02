import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder


def get_column_distribution(df, column):
  return df[column].value_counts(normalize = True)


def missing_vals_imputation(df, col):
    df_dist = get_column_distribution(df, col)

    df_values = df_dist.index
    df_probs = df_dist.values

    missing_df_col_values = df[col].isna()
    df.loc[missing_df_col_values, col] = np.random.default_rng(seed=0).choice(df_values, size = missing_df_col_values.sum(), p = df_probs)

    return df


def log_scale(df, col):
  df[f'{col}_log'] = np.log1p(df[col])
  df = df.drop(columns = [col])
  return df


def X_datasets_transform(X_train, X_test):
    
    X_train, X_test = X_fill_missing_values(X_train, X_test)
    X_train, X_test = X_feature_engineering(X_train, X_test)
    X_train, X_test = X_encode(X_train, X_test)

    return X_train, X_test


def df_transform(df):

    mapping = {'male' : 1, 'female' : 0}
    df['Is_Male'] = df['Sex'].map(mapping)
    
    df['Deck'] = df['Cabin'].str[0]

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1

    pclass_3_male_alone = df[(df['Pclass'] == 3) & (df['Parch'] == 0) & (df['Sex'] == 'male')]
    pclass_3_other = df[(df['Pclass'] == 3) & ~(df.index.isin(pclass_3_male_alone.index))]
    pclass_1 = df[(df['Pclass'] == 1)]

    for i in df.index:
        if i in pclass_3_male_alone.index:
            df.loc[i, 'Section'] = 'bow'
        elif i in pclass_3_other.index:
            df.loc[i, 'Section'] = 'stern'
        elif i in pclass_1.index:
            df.loc[i, 'Section'] = 'midship'
        else:
            df.loc[i, 'Section'] = 'midship_aft' 

    df.drop(columns = ['Sex', 'Cabin', 'SibSp', 'Parch'], inplace = True)

    return df


def X_fill_missing_values(X_train, X_test):

    fare_medians_by_class = X_train.groupby('Pclass')['Fare'].median().to_dict()
    age_medians_by_class = X_train.groupby('Pclass')['Age'].median().to_dict()
    embarked_mode = X_train['Embarked'].mode()[0]

    for df in [X_train, X_test]:

        df["Age"] = df["Age"].fillna(df["Pclass"].map(age_medians_by_class))
        df["Fare"] = df["Fare"].fillna(df["Pclass"].map(fare_medians_by_class))
        df['Embarked'] = df['Embarked'].fillna(embarked_mode)
        df = missing_vals_imputation(X_train, 'Deck')

    return X_train, X_test


def X_feature_engineering(X_train, X_test):

    is_ticket_duplicated = X_train.Ticket.duplicated()
    ticket_duplicates = X_train.loc[is_ticket_duplicated, 'Ticket'].unique()
    people_on_ticket = X_train[X_train['Ticket'].isin(ticket_duplicates)]['Ticket'].value_counts()

    for df in [X_train, X_test]:

        df['People_On_Ticket'] = df['Ticket'].map(people_on_ticket).fillna(1).astype('int')
        df['Fare_per_person'] = df['Fare'] / df['People_On_Ticket']
        df.drop(columns = ['Fare', 'People_On_Ticket', 'Ticket'], inplace = True)

    Q1 = X_train['Fare_per_person'].quantile(0.25)
    Q3 = X_train['Fare_per_person'].quantile(0.75)
    IQR = Q3-Q1
    upper_bound = round(Q3 + (1.5 * IQR), 2)

    for df in [X_train, X_test]:

        for i in df.index:
            if df.loc[i, 'Fare_per_person'] >= 200:
                df.loc[i, 'Wealth_level'] = 'elite'
            elif (df.loc[i, 'Fare_per_person'] >= upper_bound) & (df.loc[i, 'Fare_per_person'] < 200):
                df.loc[i, 'Wealth_level'] = 'wealthy'
            elif (df.loc[i, 'Fare_per_person'] >= 30) & (df.loc[i, 'Fare_per_person'] < upper_bound):
                df.loc[i, 'Wealth_level'] = 'upper_class'
            elif (df.loc[i, 'Fare_per_person'] >= 12) & (df.loc[i, 'Fare_per_person'] < 30):
                df.loc[i, 'Wealth_level'] = 'middle_class'
            else:
                df.loc[i, 'Wealth_level'] = 'lower_class'

    return X_train, X_test


def X_encode(X_train, X_test):

    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    X_train_encoded = encoder.fit_transform(X_train[['Section', 'Embarked']])
    X_test_encoded = encoder.transform(X_test[['Section', 'Embarked']])

    X_train = pd.concat([X_train.drop(columns=['Section', 'Embarked']), X_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=['Section', 'Embarked']), X_test_encoded], axis=1)


    mapping = {
        'elite': 1,
        'wealthy': 2,
        'upper_class': 3,
        'middle_class': 4,
        'lower_class': 5,
    }

    labels_mapping = {
            k: i for i, k in enumerate(sorted(X_train['Deck'].unique()))
    }

    age_min = X_train['Age'].min()
    age_max = X_train['Age'].max()

    dfs = []

    for df in [X_train, X_test]:

        df = df.copy()

        df['Wealth_level'] = df['Wealth_level'].map(mapping)
        df['Deck'] = df['Deck'].map(labels_mapping).fillna(-1)

        df['Age_normalized'] = (df['Age'] - age_min) / (age_max - age_min)
        df.drop(columns = ['Age'], inplace = True)

        df = log_scale(df, 'Fare_per_person')

        dfs.append(df)

    X_train, X_test = dfs[0], dfs[1]

    return X_train, X_test


def plot_confusion_matrix(y_test, y_pred):

    plt.figure(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
