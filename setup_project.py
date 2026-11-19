# setup_project.py (v4) - SU DUNG FILE CSV TINH & GHI DE KET QUA
# Chay tu thu muc TaxiDW_Project (ngang hang voi "project" va "venv"):
#     python setup_project.py

import os
import shutil

BASE_DIR = os.path.join(os.getcwd(), "project")
files = {}

# Xoa cac file cu khong can thiet (SQL, run_id)
files_to_remove = [
    "shared/db.py",
    "shared/sample_query.sql",
    "shared/runctx.py"
]

files["shared/__init__.py"] = r'''\n'''
files["nhanh_A_lightgbm/__init__.py"] = r'''\n'''
files["nhanh_B_randomforest/__init__.py"] = r'''\n'''

files["requirements.txt"] = r'''
pandas
numpy
scikit-learn
lightgbm
pyarrow
matplotlib
'''

# ----------------------------------------------------------------------------
files["shared/config.py"] = r'''
# CAU HINH CHUNG - CA NHOM DUNG CHUNG
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_SEED = 42                 # Co dinh de chia train/valid/test va train model giong het nhau
TARGET = 'total_amount'          # = FareAmount + SurchargeAmount (KHONG gom tip)

CSV_COLUMNS = ['trip_distance', 'PULocationID', 'DOLocationID', 'Hour', 'Weekday',
               'PeakHour', 'RatecodeID', 'passenger_count', 'total_amount']
FEATURES = [c for c in CSV_COLUMNS if c != TARGET]
CATEGORICAL_COLS = ['PULocationID', 'DOLocationID', 'RatecodeID']

# ---- NGUON DU LIEU ----
PATH_CSV_RAW = os.path.join(ROOT, 'data', 'raw', 'ml_dataset_total_amount.csv')
PATH_TRAIN = os.path.join(ROOT, 'data', 'train.parquet')
PATH_VALID = os.path.join(ROOT, 'data', 'valid.parquet')
PATH_TEST = os.path.join(ROOT, 'data', 'test.parquet')

TEST_SIZE = 0.2
VALID_SIZE = 0.1

# Luat lam sach phia Python (lop bao ve thu hai sau SQL)
CLEANING_RULES = {
    'min_total_amount': 0.01, 'max_total_amount': 500.0,
    'min_trip_distance': 0.01, 'max_trip_distance': 100.0,
    'min_passenger_count': 1, 'max_passenger_count': 6,
}

# Sieu tham so (ghi nguyen bang nay vao bao cao)
RF_PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_SEED)
LGBM_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                   importance_type='gain', n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
EARLY_STOPPING_ROUNDS = 50

# Duong dan
REPORTS_DIR = os.path.join(ROOT, 'reports')
'''

# ----------------------------------------------------------------------------
files["shared/preprocessing.py"] = r'''
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
'''

# ----------------------------------------------------------------------------
files["shared/data_io.py"] = r'''
import os
import numpy as np
import pandas as pd
from shared.config import *

def load_splits(categorical=False):
    # Doc tu file tinh, khong dung run_id
    parts = {
        'train': pd.read_parquet(PATH_TRAIN),
        'valid': pd.read_parquet(PATH_VALID),
        'test': pd.read_parquet(PATH_TEST)
    }
    out = {n: (p[FEATURES].copy(), p[TARGET].copy()) for n, p in parts.items()}
    
    if categorical:
        for col in CATEGORICAL_COLS:
            cats = np.sort(pd.concat([p[col] for p in parts.values()]).unique())
            for n in out:
                out[n][0][col] = pd.Categorical(out[n][0][col], categories=cats)
    return out
'''

# ----------------------------------------------------------------------------
files["shared/evaluation_harness.py"] = r'''
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from shared.config import *

def evaluate_model(model, X_valid, y_valid, X_test, y_test, model_name,
                   train_time, n_train, params_text=''):
    mae_valid = mean_absolute_error(y_valid, model.predict(X_valid))
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0
    mse = mean_squared_error(y_test, y_pred)
    
    result = {
        'model': model_name,
        'MAE': round(float(mean_absolute_error(y_test, y_pred)), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(np.sqrt(mse)), 4),
        'R2': round(float(r2_score(y_test, y_pred)), 4),
        'MAE_valid': round(float(mae_valid), 4),
        'training_time_seconds': round(float(train_time), 2),
        'inference_time_seconds': round(float(inference_time), 2),
        'n_train': int(n_train),
        'n_test': int(len(y_test)),
        'params': params_text,
    }
    print(f"\n--- KET QUA {model_name.upper()} ---")
    for k, v in result.items():
        print(f"{k:25s}: {v}")
    return result

def save_result(result, name):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f'results_{name}.csv')
    pd.DataFrame([result]).to_csv(path, index=False)
    print(f"Da GHI DE ket qua vao {path}")

def save_feature_importance(model, feature_names, name):
    imp = np.asarray(model.feature_importances_, dtype=float)
    df = pd.DataFrame({'feature': feature_names, 'importance_share': imp / imp.sum()})
    df = df.sort_values('importance_share', ascending=False)
    path = os.path.join(REPORTS_DIR, f'feature_importance_{name}.csv')
    df.to_csv(path, index=False)
'''

# ----------------------------------------------------------------------------
files["shared/compare.py"] = r'''
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.config import *

METRICS = ['MAE', 'RMSE', 'R2', 'MAE_valid', 'training_time_seconds', 'inference_time_seconds']

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = [p for p in os.listdir(REPORTS_DIR) if p.startswith('results_') and p != 'results_comparison.csv' and p.endswith('.csv')]
    
    if len(paths) < 2:
        print(f"Moi co {len(paths)} file ket qua. Can chay ca 2 mo hinh truoc khi so sanh.")
        return

    df = pd.concat([pd.read_csv(os.path.join(REPORTS_DIR, p)) for p in sorted(paths)], ignore_index=True)
    
    # GHI DE file so sanh
    out_path = os.path.join(REPORTS_DIR, 'results_comparison.csv')
    df.to_csv(out_path, index=False)
    
    print("\nBẢNG ĐIỂM TỔNG HỢP (GHI ĐÈ):")
    print(df[['model'] + METRICS].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(axes, ['MAE', 'RMSE', 'training_time_seconds'],
                              ['MAE (USD, thap hon la tot hon)', 'RMSE (USD, thap hon la tot hon)', 'Thoi gian huan luyen (giay)']):
        bars = ax.bar(df['model'], df[col])
        ax.set_title(title)
        ax.bar_label(bars, fmt='%.2f')
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'comparison_chart.png'), dpi=200)
    plt.close()

    lines, base = [], df[df['model'] == 'RandomForest']
    if not base.empty:
        b = base.iloc[0]
        for _, r in df[df['model'] != 'RandomForest'].iterrows():
            d_mae = (r['MAE'] - b['MAE']) / b['MAE'] * 100
            d_rmse = (r['RMSE'] - b['RMSE']) / b['RMSE'] * 100
            ratio = b['training_time_seconds'] / max(r['training_time_seconds'], 1e-9)
            lines.append(f"{r['model']} so voi RandomForest: MAE {abs(d_mae):.2f}% {'thap hon' if d_mae < 0 else 'cao hon'}, "
                         f"RMSE {abs(d_rmse):.2f}% {'thap hon' if d_rmse < 0 else 'cao hon'}, "
                         f"thoi gian huan luyen {'nhanh hon' if ratio > 1 else 'cham hon'} {max(ratio, 1 / ratio):.2f} lan.")
    
    with open(os.path.join(REPORTS_DIR, 'comparison_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nDa GHI DE bieu do va bang vao {REPORTS_DIR}")

if __name__ == '__main__':
    main()
'''

# ----------------------------------------------------------------------------
files["nhanh_A_lightgbm/run_lightgbm.py"] = r'''
import sys
import time
from shared.config import *
from shared.data_io import load_splits
from shared.evaluation_harness import evaluate_model, save_result, save_feature_importance

def run(categorical, model_name, file_name):
    import lightgbm as lgb
    print(f"\n=== {model_name} ===")
    d = load_splits(categorical=categorical)
    X_train, y_train = d['train']
    X_valid, y_valid = d['valid']
    X_test, y_test = d['test']

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='l1',
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    train_time = time.perf_counter() - t0

    params_text = (f"num_leaves={LGBM_PARAMS['num_leaves']}; lr={LGBM_PARAMS['learning_rate']}; "
                   f"best_iteration={model.best_iteration_}; categorical={categorical}")
    result = evaluate_model(model, X_valid, y_valid, X_test, y_test, model_name,
                            train_time, len(y_train), params_text)
    save_result(result, file_name)
    save_feature_importance(model, FEATURES, file_name)
    return result

if __name__ == '__main__':
    run(False, 'LightGBM', 'lightgbm')
    if '--native' in sys.argv:
        run(True, 'LightGBM (native category)', 'lightgbm_native_category')
'''

files["nhanh_B_randomforest/run_randomforest.py"] = r'''
import time
from sklearn.ensemble import RandomForestRegressor
from shared.config import *
from shared.data_io import load_splits
from shared.evaluation_harness import evaluate_model, save_result, save_feature_importance

def run():
    print("\n=== RANDOM FOREST ===")
    d = load_splits(categorical=False)
    X_train, y_train = d['train']
    X_valid, y_valid = d['valid']
    X_test, y_test = d['test']

    model = RandomForestRegressor(**RF_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    params_text = (f"n_estimators={RF_PARAMS['n_estimators']}; max_depth={RF_PARAMS['max_depth']}; "
                   f"min_samples_leaf={RF_PARAMS['min_samples_leaf']}")
    result = evaluate_model(model, X_valid, y_valid, X_test, y_test, 'RandomForest',
                            train_time, len(y_train), params_text)
    save_result(result, 'randomforest')
    save_feature_importance(model, FEATURES, 'randomforest')
    return result

if __name__ == '__main__':
    run()
'''

# ----------------------------------------------------------------------------
files["shared/run_all.py"] = r'''
import argparse
from shared import preprocessing, compare
from nhanh_A_lightgbm import run_lightgbm as lgbm_branch
from nhanh_B_randomforest import run_randomforest as rf_branch

def main():
    print("########## BAT DAU CHAY TOAN BO PIPELINE ##########")
    preprocessing.main()
    rf_branch.run()
    try:
        lgbm_branch.run(False, 'LightGBM', 'lightgbm')
    except ImportError as e:
        print(f"!!! BO QUA LightGBM vi chua cai thu vien: {e}")
    compare.main()
    print("\n########## HOAN TAT ##########")

if __name__ == '__main__':
    main()
'''

# ----------------------------------------------------------------------------
print("Dang tao cac file code (v4 - GHI DE, KHONG SQL)...")
for rel, content in files.items():
    full = os.path.join(BASE_DIR, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if os.path.exists(full):
        shutil.copy2(full, full + ".bak")
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.strip("\n") + "\n")
    print("  +", rel)

# Xoa cac file cu khong dung
for rel in files_to_remove:
    full = os.path.join(BASE_DIR, rel)
    if os.path.exists(full):
        os.remove(full)
        print("  - Da xoa file cu:", rel)

print("\nXong. Buoc tiep theo (trong thu muc project, da kich hoat venv):")
print("  1) python -m shared.preprocessing")
print("  2) python -m nhanh_A_lightgbm.run_lightgbm        (Ban A)")
print("     python -m nhanh_B_randomforest.run_randomforest (Ban B)")
print("  3) python -m shared.compare")