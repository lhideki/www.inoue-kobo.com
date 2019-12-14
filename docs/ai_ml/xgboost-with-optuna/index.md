# XGBoostをOptunaでパラメータチューニングする

## TL;DR

[XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html)のパラメータを[Optuna](https://github.com/optuna/optuna)でチューニングします。
ベンチマーク用データとしては[ボストン住宅価格データセット](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)を使用します。

## データ準備

[scikit-learn](https://scikit-learn.org/stable/)の`datasets`を使ってデータをロードします。
学習データとテストデータの分割は8:2です。

```python
from sklearn import datasets

features, labels = datasets.load_boston(return_X_y =True)
```

```python
from sklearn import model_selection

train_features, test_features, train_labels, test_labels = model_selection.train_test_split(features, labels, test_size=0.2)
```

```python
print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)
```

```
>    (404, 13)
>    (404,)
>    (102, 13)
>    (102,)
```

```python
import xgboost as xgb

trains = xgb.DMatrix(train_features, label=train_labels)
tests = xgb.DMatrix(test_features, label=test_labels)
```

## ハイパーパラメータ最適化

Optunaでパラメータチューニングを行います。チューニング対象は以下としています。

* eta・・・学習率
* max_depth・・・木の深さ
* lambda・・・L2正則化項のペナルティ

目的はR2(決定係数)を使用します。R2は大きい方が性能が高いことを表すため`direction`を`maximize`にしています。

```python
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,
    'eta': 0.01,
}

watchlist = [(trains, 'train'), (tests, 'eval')]
```

```python
import optuna
from sklearn.metrics import r2_score

def optimizer(trial):
    #booster = trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear'])
    eta = trial.suggest_uniform('eta', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 4, 15)
    __lambda = trial.suggest_uniform('lambda', 0.7, 2)

    #params['booster'] = booster
    params['eta'] = eta
    params['max_depth'] = max_depth
    params['lambda'] = __lambda

    model = xgb.train(params, trains, num_boost_round=50)
    predicts = model.predict(tests)

    r2 = r2_score(test_labels, predicts)
    print(f'#{trial.number}, Result: {r2}, {trial.params}')

    return r2
```

```python
study = optuna.create_study(direction='maximize')
study.optimize(optimizer, n_trials=500)
```

```
>    #0, Result: 0.4407700758314629, {'eta': 0.02815947490006805, 'max_depth': 15, 'lambda': 1.7745357854073418}
>
>
>    [I 2019-12-14 21:55:06,271] Finished trial#0 resulted in value: 0.4407700758314629. Current best value is 0.4407700758314629 with parameters: {'eta': 0.02815947490006805, 'max_depth': 15, 'lambda': 1.7745357854073418}.
>
>
>    #1, Result: 0.9411376545146801, {'eta': 0.22645720401324948, 'max_depth': 14, 'lambda': 0.8344469058001295}
>
>
>    [I 2019-12-14 21:55:06,399] Finished trial#1 resulted in value: 0.9411376545146801. Current best value is 0.9411376545146801 with parameters: {'eta': 0.22645720401324948, 'max_depth': 14, 'lambda': 0.8344469058001295}.
>
>
>    #2, Result: 0.9334222868385071, {'eta': 0.13902339239669845, 'max_depth': 5, 'lambda': 1.7072931624988712}
>
>
>    [I 2019-12-14 21:55:06,482] Finished trial#2 resulted in value: 0.9334222868385071. Current best value is 0.9411376545146801 with parameters: {'eta': 0.22645720401324948, 'max_depth': 14, 'lambda': 0.8344469058001295}.
>
>
>    #3, Result: 0.73803703148582, {'eta': 0.03931804573356067, 'max_depth': 14, 'lambda': 1.3872293369119229}
>
>
>    [I 2019-12-14 21:55:06,581] Finished trial#3 resulted in value: 0.73803703148582. Current best value is 0.9411376545146801 with parameters: {'eta': 0.22645720401324948, 'max_depth': 14, 'lambda': 0.8344469058001295}.
>
>
[省略]
>
>    #499, Result: 0.9451385721692452, {'eta': 0.1399512997944577, 'max_depth': 6, 'lambda': 0.9998555419185053}
>
>
>    [I 2019-12-14 21:56:24,897] Finished trial#499 resulted in value: 0.9451385721692452. Current best value is 0.9530448799539337 with parameters: {'eta': 0.15359982254807167, 'max_depth': 5, 'lambda': 1.0040675732276585}.
```

以下が探索された中でベストのパラメータです。

```python
study.best_params
```

```
>    {'eta': 0.15359982254807167, 'max_depth': 5, 'lambda': 1.0040675732276585}
```

以下はSeabornのPairplotで表示したパラメータ間の相関です。

```python
%matplotlib inline
import seaborn as sns

study_df = study.trials_dataframe()[['value', 'params']]
sns.pairplot(study_df, kind='reg')
```

![png](images/output_10_1.png)

## 見つけたパラメータでモデル作成

最適化されたパラメータを使用して学習します。

```python
params = dict(params, **study.best_params)
watchlist = [(trains, 'train'), (tests, 'eval')]

model = xgb.train(params, trains, num_boost_round=100, verbose_eval=True, evals=watchlist)
```

```
>    [0]	train-rmse:20.3573	eval-rmse:20.5125
>    [1]	train-rmse:17.4151	eval-rmse:17.5057
>    [2]	train-rmse:14.9285	eval-rmse:14.9457
>    [3]	train-rmse:12.8167	eval-rmse:12.8562
[省略]
>    [99]	train-rmse:0.346584	eval-rmse:2.04457
```

```python
predicts = model.predict(tests)
```

```python
from sklearn.metrics import r2_score

r2_score(test_labels, predicts)
```

```
>    0.9519996238263829
```

## まとめ

最終的に`0.95`と高い精度で説明可能なモデルが探索できました。

## 参考文献

* [XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html)
* [Optuna](https://github.com/optuna/optuna)