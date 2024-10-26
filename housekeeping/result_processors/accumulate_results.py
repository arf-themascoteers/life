import pandas as pd
import os


def accumulate_results(sources, dest, excluded = None):
    if excluded is None:
        excluded = []
    source_folder = "../../saved_results"
    dest_folder = "../../acc_results"

    sources = [os.path.join(source_folder, source) for source in sources if source not in excluded]
    dest = os.path.join(dest_folder, dest)

    summaries = []

    for source in sources:
        summaries = summaries + [os.path.join(source, file) for file in os.listdir(source) if file.endswith("_summary.csv")]

    os.makedirs(dest_folder, exist_ok=True)

    df = [sanitize_df(pd.read_csv(loc)) for loc in summaries]
    df = [d for d in df if len(d) != 0]
    df = pd.concat(df, axis=0, ignore_index=True)
    result_df = df.groupby(["dataset","target_size","algorithm"]).agg(
        {
            'time': 'mean',
            'oa': 'mean',
            'aa': 'mean',
            'k': 'mean',
            'selected_bands': lambda x: "---".join(x),
            'selected_weights': lambda x: "---".join(x)
        }
    ).reset_index()
    result_df.to_csv(dest, index=False)


def sanitize_df(df):
    if "algorithm" not in df.columns:
        df['target_size'] = 0
        df['algorithm'] = 'all'
        df['time'] = 0
        df['selected_bands'] = ''
        df['selected_weights'] = ''
    return df


if __name__ == "__main__":
    accumulate_results(["1","2"],"test.csv")