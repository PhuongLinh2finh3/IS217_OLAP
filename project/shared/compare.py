import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared.config import *

METRICS = ['MAE', 'RMSE', 'R2', 'MAE_valid', 'training_time_seconds', 'inference_time_seconds']

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = [p for p in os.listdir(REPORTS_DIR) if p.startswith('results_') and p != 'results_comparison.csv' and p.endswith('.csv')]
    
    if len(paths) < 2:
        print(f"Moi co {len(paths)} file ket qua. Can chay ca 2 mo hinh truoc khi so sanh.")
        return

    df = pd.concat([pd.read_csv(os.path.join(REPORTS_DIR, p)) for p in sorted(paths)], ignore_index=True)
    
    # GHI DE file so sanh
    out_path = os.path.join(REPORTS_DIR, 'results_comparison.csv')
    df.to_csv(out_path, index=False)
    
    print("\nBẢNG ĐIỂM TỔNG HỢP (GHI ĐÈ):")
    print(df[['model'] + METRICS].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(axes, ['MAE', 'RMSE', 'training_time_seconds'],
                              ['MAE (USD, thap hon la tot hon)', 'RMSE (USD, thap hon la tot hon)', 'Thoi gian huan luyen (giay)']):
        bars = ax.bar(df['model'], df[col])
        ax.set_title(title)
        ax.bar_label(bars, fmt='%.2f')
        ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'comparison_chart.png'), dpi=200)
    plt.close()

    lines, base = [], df[df['model'] == 'RandomForest']
    if not base.empty:
        b = base.iloc[0]
        for _, r in df[df['model'] != 'RandomForest'].iterrows():
            d_mae = (r['MAE'] - b['MAE']) / b['MAE'] * 100
            d_rmse = (r['RMSE'] - b['RMSE']) / b['RMSE'] * 100
            ratio = b['training_time_seconds'] / max(r['training_time_seconds'], 1e-9)
            lines.append(f"{r['model']} so voi RandomForest: MAE {abs(d_mae):.2f}% {'thap hon' if d_mae < 0 else 'cao hon'}, "
                         f"RMSE {abs(d_rmse):.2f}% {'thap hon' if d_rmse < 0 else 'cao hon'}, "
                         f"thoi gian huan luyen {'nhanh hon' if ratio > 1 else 'cham hon'} {max(ratio, 1 / ratio):.2f} lan.")
    
    with open(os.path.join(REPORTS_DIR, 'comparison_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nDa GHI DE bieu do va bang vao {REPORTS_DIR}")

if __name__ == '__main__':
    main()
