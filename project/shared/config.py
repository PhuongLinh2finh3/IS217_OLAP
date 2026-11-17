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
