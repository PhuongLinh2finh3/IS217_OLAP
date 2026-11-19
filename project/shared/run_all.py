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
