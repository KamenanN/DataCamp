import pandas as pd
from sklearn import model_selection

def fold(df, cross_val= model_selection.KFold(n_splits=5)):

    #we create a new column called kfold and fill it with -1
    df['kfold'] = -1

    #the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fill the new kfold column
    for fold, (t_, v_) in enumerate(model.split(X=df)):
        df.loc[v_, "kfold"] = fold
    # save a new csv with a kfold column
    df = df.to_csv("df_train_folds.csv", index=False)
    return df




