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
