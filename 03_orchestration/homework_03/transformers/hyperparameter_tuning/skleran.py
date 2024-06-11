import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt
#from mlops.homework03.utils.models.sklearn import load_class, train_model


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer



def transform(data, *args, **kwargs):
    print(data_2)
    train_dicts = data[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = data[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    print(f'Train RMSE: {mean_squared_error(y_train, y_pred, squared=False)}')
    """
    Template code for a transformer block.
    """
    # Specify your transformation logic here

    return lr, lr_info


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'