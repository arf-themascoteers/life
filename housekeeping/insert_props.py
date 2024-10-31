import pandas as pd
import os

for folder in os.listdir("../saved_results"):
    loc = os.path.join("../saved_results", folder)
    for f in os.listdir(loc):
        if ("all_features_details" in f) or ("all_features_summary" in f):
            continue
        csv_loc = os.path.join(loc, f)
        print(csv_loc)
        df = pd.read_csv(csv_loc)
        df.insert(3,"props",0)
        df.to_csv(csv_loc, index = False)