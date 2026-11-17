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

    print(f"\n--- KẾT QUẢ {model_name.upper()} ---")
    for k, v in result.items():
        print(f"{k:25s}: {v}")
    return result

def save_result(result, path):
    pd.DataFrame([result]).to_csv(path, index=False)
    print(f"✅ Đã lưu kết quả vào {path}")
