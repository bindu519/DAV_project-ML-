"""
train_models.py
---------------
Run this once:   python train_models.py
It cleans the data, trains Linear Regression and Random Forest for both tasks,
prints the comparison tables and saves the trained models to models/.
(The Streamlit app will also train automatically if this file has not been run.)
"""
import time

import pandas as pd

from ml_pipeline import MODEL_FILE, TASKS, save_models, train_all
from utils import clean_data, load_raw

if __name__ == "__main__":
    t0 = time.time()
    raw = load_raw()
    df, log, checks = clean_data(raw)
    print(f"Loaded {checks['rows_raw']:,} rows -> {checks['rows_clean']:,} rows after cleaning\n")

    results = train_all(df)
    pd.options.display.float_format = "{:,.4f}".format
    for key, cfg in TASKS.items():
        r = results[key]
        print("=" * 70)
        print(cfg["label"], f"| train={r['n_train']:,} test={r['n_test']:,} | RF min_samples_leaf={r['rf_leaf']}")
        print(r["metrics"].to_string(index=False))
        print("Best model:", r["best_name"], "\n")

    save_models(results)
    print(f"Saved to {MODEL_FILE}  (took {time.time() - t0:.0f}s)")
