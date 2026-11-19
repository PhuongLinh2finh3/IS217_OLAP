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
