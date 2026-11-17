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

    print("\nBẢNG ĐIỂM TỔNG HỢP:")
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
    print("\n✅ Đã lưu biểu đồ so sánh vào reports/comparison_chart.png")

if __name__ == '__main__':
    main()
