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
