"""
XGBoost is an optimized distributed gradient boosting library 
designed to be highly efficient, flexible and portable. 

@author: Instructor Zhang
"""
import warnings
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


class XGBEstimator():
    """The XGBoost classification/regression model."""
    def __init__(self):
        pass

    def train(self, df, target, task):
        """
        Train (fit) the model.
        
        Parameters:
            df: pandas dataframe, processed training dataset 
            target: string, name of target column
            task: string, either 'classification' or 'regression'
            
        Return:
            A fitted XGBoost model
        """
        train_x = df.drop(labels=[target], axis=1)
        train_y = df[target]

        hyper_param = {
            "max_depth": [3, 6],
            "n_estimators": [100, 150],
            "booster": ["gbtree", "gblinear", "dart"],
            "subsample": [0.8, 1.0]
        }
        
        if task == 'classification':
            model = XGBClassifier(random_state=42, n_jobs=8, verbosity=0)
            scoring = 'accuracy'
        else:
            model = XGBRegressor(random_state=42, n_jobs=8, verbosity=0)
            scoring = 'r2'
        
        gs = GridSearchCV(
            estimator=model,
            param_grid=hyper_param,
            scoring=scoring,
            iid=False, cv=5, n_jobs=-1, verbose=1
        )        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(train_x, train_y)

        print("Best parameters: {0}".format(gs.best_params_))
        return gs.best_estimator_
    
    
    def evaluation(self, model, test_df, target, task):
        """Show various evaluation results based on sklearn's scheme."""
        prediction = model.predict(test_df.drop(target, axis=1))
        y_true = test_df[target]
        
        res = {}
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(test_df.drop(target, axis=1))
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(test_df.drop(target, axis=1))
        
        if task == 'classification':
            f1 = metrics.f1_score(y_true, prediction, average="macro")
            accuracy = metrics.accuracy_score(y_true, prediction)
            print('f1-score: {0:f}'.format(f1))
            print('Accuracy: {0:f}'.format(accuracy))
            
            res['f1_score'] = f1
            res['accuracy'] = accuracy
            res['probability_prediction'] = y_score.tolist()
        else:
            r2 = metrics.r2_score(y_true, prediction)
            mse = metrics.mean_squared_error(y_true, prediction)
            print('r2 score: {0:f}'.format(r2))
            print('Mean-squared error: {0:f}'.format(mse))            

            res['r2_score'] = r2
            res['mean_squared_error'] = mse
            
        return res


