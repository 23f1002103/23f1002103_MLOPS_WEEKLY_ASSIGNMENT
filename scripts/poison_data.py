import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
INPUT_FILE = "data/v1data.csv"
OUTPUT_DIR = "data"
POISON_LEVELS = [5, 10, 50]
LABEL_COL = "species"

def poison_dataset(df, corruption_pct, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    poisoned_df = df.copy()
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    n_rows = len(df)
    n_poison = int(round(n_rows * corruption_pct / 100))
    poison_idx = rng.choice(df.index, size=n_poison, replace=False)
    feature_ranges = {col: (df[col].min(), df[col].max()) for col in feature_cols}
    unique_labels = df[LABEL_COL].unique()
    for idx in poison_idx:
        for col in feature_cols:
            low, high = feature_ranges[col]
            poisoned_df.at[idx, col] = rng.uniform(low, high)
        poisoned_df.at[idx, LABEL_COL] = rng.choice(unique_labels)
    return poisoned_df

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {INPUT_FILE} ({len(df)} rows)")

    clean_path = os.path.join(OUTPUT_DIR, "iris_clean.csv")
    df.to_csv(clean_path, index=False)
    print(f"Saved clean copy -> {clean_path}")

    for pct in POISON_LEVELS:
        poisoned_df = poison_dataset(df, pct)
        out_path = os.path.join(OUTPUT_DIR, f"iris_poisoned_{pct}.csv")
        poisoned_df.to_csv(out_path, index=False)
        n_poison = int(round(len(df) * pct / 100))
        print(f"Saved {pct}% poisoned dataset -> {out_path} ({n_poison} of {len(df)} rows corrupted)")

if __name__ == "__main__":
    main()
