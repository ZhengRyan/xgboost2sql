#!/usr/bin/env python

"""Tests for `xgboost2sql` package."""

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from xgboost2sql import XGBoost2Sql

X, y = make_classification(n_samples=10000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=2,
                           n_repeated=0,
                           n_classes=2,
                           weights=[0.7, 0.3],
                           flip_y=0.1,
                           random_state=1024)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1024)

###训练模型
model = xgb.XGBClassifier(n_estimators=3)
model.fit(X_train, y_train)
xgb.to_graphviz(model)

###使用xgboost2sql包将模型转换成的sql语句
xgb2sql = XGBoost2Sql()
sql_str = xgb2sql.transform(model)
print(sql_str)
xgb2sql.save()
