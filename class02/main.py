# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 12:31 上午
# @Author  : Ian
# @File    : main.py
# @Project : class02

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from xgboost import plot_importance


import sklearn

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix, f1_score, recall_score, roc_curve
from sklearn.metrics import auc
import sklearn.metrics as metrics

train_path = "data/train_magnetic.csv"

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

# 设置中文显示问题
plt.rcParams['font.sans-serif'] = ['Songti SC']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False




# 三分类
def generate_dataset():
    df = pd.read_csv(train_path)

    #df['is_magnetic'] = df['is_magnetic'].astype('int64')

    #计算种类数目
    n_classes = df['thermodynamic_stability_level'].nunique()

    le_thermodynamic = LabelEncoder()
    df['thermodynamic_encoded'] = le_thermodynamic.fit_transform(df['thermodynamic_stability_level'])

    df.drop(['materials', 'is_magnetic','thermodynamic_stability_level'], axis=1, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True,
                                         stratify=df['thermodynamic_encoded'],
                                         random_state=33)

    y_train = train_df['thermodynamic_encoded'].values
    y_test = test_df['thermodynamic_encoded'].values

    X_train = train_df.iloc[:,0:-1].values
    X_test = test_df.iloc[:, 0:-1].values

    feature_names = train_df.iloc[:, 0:-1].columns.tolist()

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, n_classes, feature_names

def draw_roc(y_test, y_score, n_classes):
    y_test_lb = LabelBinarizer().fit_transform(y_test)

    tpr = [[] for _ in range(n_classes)]
    fpr = [[] for _ in range(n_classes)]
    roc_auc = [[] for _ in range(n_classes)]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_lb[:,i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.savefig('roc曲线.png')
    plt.show()


def param_grid_search(model,param_grid, X_train, y_train):

    print('------------param_grid_search--------------')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    clf = GridSearchCV(estimator=model, param_grid = param_grid, scoring='f1_macro', cv=cv, n_jobs=2)
    clf.fit(X_train, y_train)

    print(clf.best_estimator_)
    print(clf.cv_results_['mean_test_score'])
    print(clf.best_params_)
    print(clf.best_score_)

    return clf.best_estimator_


def train_predict(model, X_train, y_train, X_test, y_test, feature_names):

    print('------------train_predict--------------')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))

    eval_res = dict()

    eval_res['acc'] = metrics.accuracy_score(y_test, y_pred)
    eval_res['pre'] = metrics.precision_score(y_test, y_pred, average='macro')
    eval_res['rec'] = metrics.recall_score(y_test, y_pred, average='macro')
    eval_res['f1'] = metrics.f1_score(y_test, y_pred, average='macro')

    eval_df = pd.DataFrame(eval_res, index = [0])
    eval_df.to_csv('eval_res')

    draw_roc(y_test, y_score, n_classes)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_importance(model, max_num_features=10,
                    ax=ax, height=0.4, importance_type='gain').set_yticklabels(feature_names)
    #plt.savefig('feature_importance.png')
    plt.show()



if __name__ == '__main__':

    params = {'booster': 'gbtree',
              'max_depth': 5,
              'learning_rate': 0.15,
              'n_estimators': 500,
              'minchildweight': 1,
              'colsample_bytree': 0.8,
              'gamma': 0,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'objective':'multi:softmax'}

    X_train, y_train, X_test, y_test, n_classes, feature_names = generate_dataset()

    params['num_class'] = n_classes

    model = xgb.XGBClassifier(**params)

    param_grid1 = {
        "learning_rate": [0.15, 0.1, 0.05],
        "max_depth": range(5, 20, 5),
        "n_estimators": range(300, 500, 50)
    }
    model = param_grid_search(model, param_grid1, X_train, y_train)

    train_predict(model, X_train, y_train, X_test, y_test, feature_names)



    print('--------')





