import pandas as pd
from pathlib import Path

FEATURE_SOURCE = Path("data/features/karachi_features_v2.csv")
FEATURE_STORE_DIR = Path("data/feature_store")
FEATURE_STORE_FILE = FEATURE_STORE_DIR / "karachi_feature_store.parquet"


def create_feature_store():
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_SOURCE)

    # Ensure timestamp is properly stored
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Save versioned feature store
    df.to_parquet(FEATURE_STORE_FILE, index=False)

    print(f"Feature Store created: {FEATURE_STORE_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns)}")


def load_features():
    return pd.read_parquet(FEATURE_STORE_FILE)


if __name__ == "__main__":
    create_feature_store()