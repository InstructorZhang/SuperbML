"""
The main function to run different models.

@author: Instructor Zhang
"""
import argparse
import pandas as pd

from awesome_lightgbm import LightGBMEstimator
from awesome_sgd import SGDEstimator
from awesome_xgboost import XGBEstimator
import data_processor


def run(path, model_obj, target, task):
    """
    Parameter:
        path: string, the file path of the csv dataset
        model_obj: object of a model 
        target: string, the name of the target column
        task: string, either classification or regrssion
        
    Return:
        A dictionary of evaluation results.
    """
    # currently only support .csv file
    ds = pd.read_csv(path)
    # drop those rows where the target value is missing
    ds.dropna(subset=[target], inplace=True)
    
    # pre-process the dataset
    trans_ds = data_processor.preprocessor(ds, target)
    
    # split the dataset into training and test sets
    train_df, test_df = data_processor.splitter(trans_ds, target, task)
    
    # train and get the best model through hyper-parameter search
    estimator = model_obj.train(train_df, target, task)
    evaluator = model_obj.evaluation(estimator, test_df, target, task)

    return evaluator



if __name__ == '__main__':
    """
    Must prepare your dataset as an csv file.
    The dataset shoud not contain date/time column.
    
    Make sure you know which column the target column is.
    """
    # will add command line argument input later
    parser = argparse.ArgumentParser(
        description='Train a dataset using awesome model, and perform evaluation.'
    )
    
    #------------ modify the below three lines for specific task --------------#
    data_path = "/path/to/your/dataset/XXXXX.csv"
    target = "XXXXX"      # name of the target column
    task_type = "xxxxx"   # either "classification" or "regression"
    #--------------------------------------------------------------------------#
    
    # pass any model object into run() function
    lgbm_obj = LightGBMEstimator()
    sgd_obj = SGDEstimator()
    xgb_obj = XGBEstimator()
    
    evaluation_res = run(data_path, sgd_obj, target, task_type)
    
    
    