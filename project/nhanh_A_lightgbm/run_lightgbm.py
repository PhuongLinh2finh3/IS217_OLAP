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
