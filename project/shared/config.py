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
# ĐÃ CẬP NHẬT TÊN FILE THÀNH dataset.csv
PATH_CSV_RAW = os.path.join(ROOT, 'data', 'raw', 'dataset.csv')
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