import os
import textwrap

BASE_DIR = os.path.join(os.getcwd(), "project")

files = {
    "shared/00_config.py": '''
import os

RANDOM_SEED = 42
TARGET = 'total_amount'

# Cập nhật Features khớp với file CSV của bạn
FEATURES = [
    'trip_distance',
    'PULocationID',
    'DOLocationID',
    'Hour',
    'Weekday',
    'PeakHour',
    'RatecodeID',
    'passenger_count'
]

CATEGORICAL_COLS = ['PULocationID', 'DOLocationID', 'RatecodeID']

TEST_SIZE = 0.2
SAMPLE_SIZE = 1000000

PATH_CSV_RAW = 'data/raw/ml_dataset_total_amount.csv'
PATH_TRAIN = 'data/train.parquet'
PATH_TEST = 'data/test.parquet'

PATH_RESULT_LIGHTGBM = 'reports/results_lightgbm.csv'
PATH_RESULT_RF = 'reports/results_randomforest.csv'
PATH_RESULT_COMPARISON = 'reports/results_comparison.csv'
''',

    "shared/01_preprocessing.py": '''
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
''',

    "shared/02_evaluation_harness.py": '''
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, model_name, train_time):
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    result = {
        'model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'training_time_seconds': round(train_time, 2),
        'inference_time_seconds': round(inference_time, 2)
    }
    
    print(f"\\n--- KẾT QUẢ {model_name.upper()} ---")
    for k, v in result.items():
        print(f"{k:25s}: {v}")
    return result

def save_result(result, path):
    pd.DataFrame([result]).to_csv(path, index=False)
    print(f"✅ Đã lưu kết quả vào {path}")
''',

    "shared/03_compare.py": '''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shared.config import *

def main():
    print("=== TỔNG HỢP KẾT QUẢ ===")
    df_lgb = pd.read_csv(PATH_RESULT_LIGHTGBM)
    df_rf = pd.read_csv(PATH_RESULT_RF)
    
    df_compare = pd.concat([df_lgb, df_rf], ignore_index=True)
    df_compare.to_csv(PATH_RESULT_COMPARISON, index=False)
    
    print("\\nBẢNG ĐIỂM TỔNG HỢP:")
    print(df_compare[['model', 'MAE', 'RMSE', 'R2', 'training_time_seconds']].to_string(index=False))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_compare, x='model', y='MAE', ax=axes[0], palette=['skyblue', 'lightgreen'])
    axes[0].set_title('So sánh MAE (Thấp hơn là tốt hơn)')
    axes[0].bar_label(axes[0].containers[0])
    
    sns.barplot(data=df_compare, x='model', y='RMSE', ax=axes[1], palette=['skyblue', 'lightgreen'])
    axes[1].set_title('So sánh RMSE (Thấp hơn là tốt hơn)')
    axes[1].bar_label(axes[1].containers[0])
    
    plt.tight_layout()
    plt.savefig('reports/comparison_chart.png', dpi=300)
    print("\\n✅ Đã lưu biểu đồ so sánh vào reports/comparison_chart.png")

if __name__ == '__main__':
    main()
''',

    "shared/results_template.csv": "model,MAE,MSE,RMSE,R2,training_time_seconds,inference_time_seconds\n",

    "nhanh_A_lightgbm/run_lightgbm.py": '''
import pandas as pd
import time
import lightgbm as lgb
from shared.config import *
from shared.evaluation_harness import evaluate_model, save_result

print("=== BẮT ĐẦU CHẠY NHÁNH LIGHTGBM ===")
train_df = pd.read_parquet(PATH_TRAIN)
test_df = pd.read_parquet(PATH_TEST)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

for col in CATEGORICAL_COLS:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print("Đang train LightGBM...")
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63, 
                          n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)

t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

result = evaluate_model(model, X_test, y_test, 'LightGBM', train_time)
save_result(result, PATH_RESULT_LIGHTGBM)
''',

    "nhanh_B_randomforest/run_randomforest.py": '''
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from shared.config import *
from shared.evaluation_harness import evaluate_model, save_result

print("=== BẮT ĐẦU CHẠY NHÁNH RANDOM FOREST ===")
train_df = pd.read_parquet(PATH_TRAIN)
test_df = pd.read_parquet(PATH_TEST)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# Random Forest không hiểu kiểu 'category', phải convert về int
for col in CATEGORICAL_COLS:
    X_train[col] = X_train[col].astype('int16')
    X_test[col] = X_test[col].astype('int16')

print("Đang train Random Forest (có thể mất vài phút)...")
model = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=RANDOM_SEED)

t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

result = evaluate_model(model, X_test, y_test, 'RandomForest', train_time)
save_result(result, PATH_RESULT_RF)
'''
}

print("Đang tạo cấu trúc thư mục và các file code...")
for filepath, content in files.items():
    full_path = os.path.join(BASE_DIR, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Tự động xóa khoảng trắng thừa đầu dòng (dedent)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip() + "\n")

print("✅ Đã tạo xong toàn bộ file!")
print("👉 Bước tiếp theo: Chạy lệnh 'python -m shared.01_preprocessing' để sinh file Parquet")