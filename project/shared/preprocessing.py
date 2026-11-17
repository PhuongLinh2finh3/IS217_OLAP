import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shared.config import *

def load_and_clean_csv():
    print(f"Đang đọc dữ liệu từ {PATH_CSV_RAW}...")
    COLS = FEATURES + [TARGET]

    # Đọc file CSV không có header
    df = pd.read_csv(PATH_CSV_RAW, header=None, names=COLS, nrows=SAMPLE_SIZE)

    # Ép kiểu số
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Làm sạch: bỏ NaN, bỏ cước phí <= 0, bỏ quãng đường <= 0
    n_before = len(df)
    df = df.dropna()
    df = df[(df['total_amount'] > 0) & (df['trip_distance'] > 0)]
    print(f"  → Đã loại {n_before - len(df):,} dòng rác. Còn lại: {len(df):,} dòng")

    return df

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    df = load_and_clean_csv()

    print("Đang chia tập Train/Test (80/20)...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    train_df.to_parquet(PATH_TRAIN, index=False)
    test_df.to_parquet(PATH_TEST, index=False)

    print(f"✅ Đã lưu Train: {PATH_TRAIN} (Shape: {train_df.shape})")
    print(f"✅ Đã lưu Test:  {PATH_TEST} (Shape: {test_df.shape})")

if __name__ == '__main__':
    main()
