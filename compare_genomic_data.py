import pandas as pd
import numpy as np
from compare_pipeline import compare_performance
import logging


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # names = ["nutt-2003-v1_database", "armstrong-2002-v2_database","bredel-2005_database", "tomlins-2006_database"]
    # values = [["CG", "CO", "NG", "NO"], ["ALL", "MLL", "AML"], ["GBM", "OG", "A"], ["EPI", "MET", "PCA", "PIN", "STROMA"]]
    names = ["armstrong-2002-v2_database"]
    values = [["ALL", "MLL", "AML"]]

    for i in range(len(names)):
            k_clusters=len(values[i])
            df = pd.read_csv(f"{names[i]}.txt", sep='\t')
            df.index = np.concatenate([["label"], df["Unnamed: 0"][1:]])
            df.drop("Unnamed: 0", axis=1, inplace=True)
            df = df.T
            orig_lables = df["label"]
            lables = [-1]*len(orig_lables)
            for j in range(len(orig_lables)):
                if orig_lables[j] in values[i]:
                    lables[j] = values[i].index(orig_lables[j])
                else:
                    raise ValueError(f"Wrong labels in {names[i]}")
            df.drop("label", axis=1, inplace=True)
            df = df.astype(float)

            variables = df.columns

            compare_performance(data = df, df_numeric=df, lables=lables, k_clusters=k_clusters,
                                     name=f"{names[i]}",
                                nunber_of_features=list(range(1,len(variables), 50))+[len(variables)],
                                title=names[i], paral=True)