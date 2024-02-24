import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score, mean_squared_error
from xgboost import XGBRegressor

from emc.classifiers import Classifier
from math import isnan

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SingleGradientBoosterBayesian(Classifier):
    def _preprocess(self, data: pd.DataFrame) -> tuple[np.ndarray, np.array]:
        groups = data.groupby(['scenario', 'simulation'])

        features = {}
        targets = {}

        for key, df in groups:
            df = df.drop(columns=['simulation', 'scenario', 'time', 'ERR'])

            df = df.reset_index(drop=True)
            target = df['target'].iloc[-1]
            if isnan(target):
                continue

            del df['target']
            targets[key] = target

            row = df.to_numpy().T.flatten()
            features[key] = row

        return features, targets

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        def objective(trial):
            hyperparams = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
            }

            print(f"Trial {trial.number}: Testing with params {hyperparams}")
            
            model = XGBRegressor(**hyperparams, random_state=self.SEED, missing=np.nan)
            model.fit(X_train, y_train)

            X_test = np.vstack(tuple(self.features_test.values()))
            y_test = np.array(tuple(self.targets_test.values()))

            y_test = (y_test < 0.85).astype(int)

            preds = model.predict(X_test)
            preds = (preds < 0.85).astype(int)

            # check scores
            f1 = f1_score(y_test, preds, average='weighted')

            print(f"Trial {trial.number} F1: {f1}")

            return f1
          
        study = optuna.create_study(direction='maximize')
        print("Optimization process started...")
        study.optimize(objective, n_trials=100, timeout=600)
        print("Optimization process completed.")

        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")

        self.parameters = best_hyperparams

        self.xgb = XGBRegressor(**best_hyperparams, random_state=self.SEED, missing=np.nan)
        self.xgb.fit(X_train, y_train)
        print("Final model trained with best hyperparameters.")

    def test(self, X_test: pd.DataFrame, y_test: pd.Series = np.array([])):
        print(f"Predicting with {len(X_test)} simulations...")
        predictions = self.xgb.predict(X_test)
        return predictions