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
