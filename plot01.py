import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_smoothed_accuracy(file_path, window_size=5):
    rounds = []
    accs = []

    # 1. 提取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        current_round = None
        for line in f:
            # 匹配轮次: [Global Round X]
            round_match = re.search(r'\[Global Round (\d+)\]', line)
            if round_match:
                current_round = int(round_match.group(1))

            # 匹配准确率: Global Aggregation | ... | Test Acc: X%
            if 'Global Aggregation' in line and 'Test Acc' in line:
                acc_match = re.search(r'Test Acc:\s*([\d\.]+)%', line)
                if acc_match:
                    acc_val = float(acc_match.group(1))
                    if current_round is not None:
                        rounds.append(current_round)
                        accs.append(acc_val)

    if not rounds:
        print("未找到有效数据，请检查日志格式。")
        return

    # 转换为 DataFrame 方便处理
    df = pd.DataFrame({'Round': rounds, 'Accuracy': accs})

    # 2. 平滑处理
    # 使用中心窗口计算均值（平滑线）
    df['Smoothed'] = df['Accuracy'].rolling(window=window_size, center=True).mean().fillna(df['Accuracy'])
    # 使用中心窗口计算最大/最小值（灰色区域边界）
    df['Upper'] = df['Accuracy'].rolling(window=window_size, center=True).max().fillna(df['Accuracy'])
    df['Lower'] = df['Accuracy'].rolling(window=window_size, center=True).min().fillna(df['Accuracy'])

    # 3. 绘图
    plt.figure(figsize=(10, 6))

    # 绘制灰色阴影区域（表示波动范围）
    plt.fill_between(df['Round'], df['Lower'], df['Upper'], color='gray', alpha=0.3,
                     label=f'Range (window={window_size})')

    # 绘制平滑后的主曲线
    plt.plot(df['Round'], df['Smoothed'], color='#1f77b4', linewidth=2, label='Smoothed Accuracy')

    # 绘制原始数据点（可选）
    plt.scatter(df['Round'], df['Accuracy'], color='#1f77b4', s=10, alpha=0.3, label='Raw Data Points')

    # 图表细节
    plt.title('Accuracy vs Global Round', fontsize=14)
    plt.xlabel('Global Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 保存结果
    plt.savefig('smoothed_accuracy_curve.png', dpi=300)
    plt.show()
    print("图像已成功生成并保存。")


# 指定您的日志文件名即可运行
log_filename = 'fedavg_resnet18_layerwise_quant_20260317_011015.log'
plot_smoothed_accuracy(log_filename, window_size=5)