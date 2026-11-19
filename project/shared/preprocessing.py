# MOI LAN CHAY: Doc file CSV tinh -> lam sach -> chia train/valid/test -> GHI DE vao data/
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shared.config import *

def load_raw():
    if not os.path.exists(PATH_CSV_RAW):
        raise FileNotFoundError(f"Khong thay {PATH_CSV_RAW}. Hay dat file CSV vao dung thu muc data/raw/")
    print(f"Dang doc file: {PATH_CSV_RAW}")
    # Code doc file khong co header (header=None)
    df = pd.read_csv(PATH_CSV_RAW, header=None, names=CSV_COLUMNS)
    return df

def clean(df):
    R = CLEANING_RULES
    for c in CSV_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n0 = len(df)
    df = df.dropna()
    print(f"  [thieu/NaN]                              loai {n0 - len(df):>9,}")
    
    steps = [
        ('total_amount ngoai khoang',    lambda d: d['total_amount'].between(R['min_total_amount'], R['max_total_amount'])),
        ('trip_distance ngoai khoang',   lambda d: d['trip_distance'].between(R['min_trip_distance'], R['max_trip_distance'])),
        ('passenger_count ngoai khoang', lambda d: d['passenger_count'].between(R['min_passenger_count'], R['max_passenger_count'])),
        ('Hour ngoai 0-23',              lambda d: d['Hour'].between(0, 23)),
        ('Weekday ngoai 0-6',            lambda d: d['Weekday'].between(0, 6)),
        ('PeakHour khac 0/1',            lambda d: d['PeakHour'].isin([0, 1])),
    ]
    for name, fn in steps:
        before = len(df)
        df = df[fn(df)]
        print(f"  [{name:<34}] loai {before - len(df):>9,}")
        
    for c in ['PULocationID', 'DOLocationID', 'RatecodeID', 'Hour', 'Weekday', 'PeakHour', 'passenger_count']:
        df[c] = df[c].astype('int32')
    df['trip_distance'] = df['trip_distance'].astype('float32')
    df['total_amount'] = df['total_amount'].astype('float32')
    return df.reset_index(drop=True)

def main():
    os.makedirs(os.path.dirname(PATH_TRAIN), exist_ok=True)
    print("=== BAT DAU TIEN XU LY DU LIEU (GHI DE) ===")
    
    df = load_raw()
    print(f"  -> Nhan {len(df):,} dong tu CSV")
    
    df = clean(df)
    print(f"  -> Con lai {len(df):,} dong sau khi lam sach")

    print("Chia train / valid / test...")
    train_full, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train, valid = train_test_split(train_full, test_size=VALID_SIZE, random_state=RANDOM_SEED)
    
    train.to_parquet(PATH_TRAIN, index=False)
    valid.to_parquet(PATH_VALID, index=False)
    test.to_parquet(PATH_TEST, index=False)
    
    print(f"  -> train: {train.shape} | valid: {valid.shape} | test: {test.shape}")
    
    check = pd.DataFrame({
        'tap': ['train', 'valid', 'test'],
        'so_dong': [len(train), len(valid), len(test)],
        'total_amount_mean': [train[TARGET].mean(), valid[TARGET].mean(), test[TARGET].mean()],
        'trip_distance_mean': [train['trip_distance'].mean(), valid['trip_distance'].mean(), test['trip_distance'].mean()],
    })
    print("\nKiem tra phan phoi giua cac tap:")
    print(check.to_string(index=False))
    print(f"\nXONG. Da ghi de file parquet vao {os.path.dirname(PATH_TRAIN)}")

if __name__ == '__main__':
    main()
