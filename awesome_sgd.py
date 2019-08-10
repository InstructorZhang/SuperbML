"""
Linear model with stochastic gradient descent (SGD) training.

@author: Instructor Zhang
"""
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


class SGDEstimator():
    """The sklearn SGD classification/regression model."""
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
            A fitted SGD model
        """
        train_x = df.drop(labels=[target], axis=1)
        train_y = df[target]

        hyper_param = {
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "max_iter": [1000, 1500]
        }
        
        if task == 'classification':
            model = SGDClassifier(random_state=None, n_jobs=-1)
            hyper_param["loss"] = ['hinge', 'log', 'perceptron']
            scoring = 'accuracy'
        else:
            model = SGDRegressor(random_state=None, n_jobs=-1)
            hyper_param["loss"] = ['squared_loss', 'huber', 'epsilon_insensitive']
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

