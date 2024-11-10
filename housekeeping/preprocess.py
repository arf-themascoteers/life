import pandas as pd
import os
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder


source_folder = "../data_raw"
output_folder = "../data"


def is_common( name):
    return name in ["indian_pines.csv", "salinas.csv", "paviaU.csv"]


def is_regression( name):
    return name in ["lucas_r.csv"]


def is_special_classification( name):
    return name in ["ghisaconus.csv","ghisaconus_health.csv"]


os.makedirs(output_folder, exist_ok=True)


def preproc():
    for file in os.listdir(source_folder):
        path = os.path.join(source_folder, file)
        df = pd.read_csv(path)

        if is_regression(file):
            df[:] = minmax_scale(df.values)
        else:
            df.iloc[:, :-1] = minmax_scale(df.iloc[:, :-1])

        if is_common(file):
            df = df[df.iloc[:, -1] != 0]
            df.iloc[:, -1] = df.iloc[:, -1] - 1
        elif is_special_classification(file):
            class_column = df.columns[-1]
            le = LabelEncoder()
            df[class_column] = le.fit_transform(df[class_column])

        dest = os.path.join(output_folder, file)
        df.to_csv(dest, index=False)


preproc()