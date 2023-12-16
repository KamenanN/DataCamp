# src/train.py
import os
import joblib
import pandas as pd
from sklearn import metrics


def run(fold, target_col, model):

    #training is different of the provided fold
    df_fold = pd.read_csv(config.TRAINING_FILE)
    df_train = df_fold.loc[df_fold.kfold != fold].reset_index(drop=True)

    #Validation data is where kfold is equal to provided fold
    df_valid = df_fold.loc[df_fold.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop(target_col, axis=1).values
    y_train = df_train.target_col.values

    # Fetch the model from model_dispatcher
    mdl = model_dispatcher

    # Idem for validation 
    x_val = df_valid.drop(target_col, axis=1).values
    y_val = df_valid.label.values

    # Fit the model on training data
    mdl = model
    mdl.fit(x_train, y_train)

    # Create predictions for validation samples
    preds = mdl.predict(x_val)
     # Save model 
    joblib.dump(mdl,
                os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
               )




    