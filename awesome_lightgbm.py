"""
LightGBM is a gradient boosting framework that 
uses tree based learning algorithms.

@author: Instructor Zhang
"""
import warnings
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

    
class LightGBMEstimator():
    """The LightGBM classification/regression model."""
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
            A fitted lightGBM model
        """
        train_x = df.drop(labels=[target], axis=1)
        train_y = df[target]
        
        if task == 'classification':
            model = LGBMClassifier( 
                bagging_freq=1, bagging_fraction=0.8, subsample_freq=0,
                random_state=42, n_jobs=-1, silent=True
            )
        else:
            model = LGBMRegressor( 
                subsample_freq=1, bagging_fraction=0.8,
                random_state=42, n_jobs=-1, silent=True
            )
            
        scoring = 'accuracy' if task == 'classification' else 'r2'       
        hyper_param = {
            "num_leaves": [20, 31, 50],
            "n_estimators": [100, 300],
            "max_bin": [127, 255],
            "boosting_type": ['gbdt', 'dart']
        }
        
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
            print('F1-score: {0:f}'.format(f1))
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
