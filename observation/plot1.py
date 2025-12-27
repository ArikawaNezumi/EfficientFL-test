import matplotlib.pyplot as plt
import numpy as np

# 1. 实验参数和您提供的损失数据
NUM_ROUNDS = 30
QUANTIZATION_LEVELS = [2, 4, 8, 16]

# 存储损失数据的字典
loss_data = {
    2: [0.9398, 0.2307, 0.1314, 0.1217, 0.0882, 0.0734, 0.0693, 0.0570, 0.0554, 0.0528, 0.0516, 0.0447, 0.0431, 0.0459,
        0.0422, 0.0451, 0.0397, 0.0437, 0.0385, 0.0369, 0.0344, 0.0353, 0.0374, 0.0328, 0.0354, 0.0361, 0.0367, 0.0340,
        0.0363, 0.0332],
    4: [0.7024, 0.1778, 0.1033, 0.0986, 0.0688, 0.0681, 0.0610, 0.0556, 0.0518, 0.0485, 0.0499, 0.0480, 0.0431, 0.0409,
        0.0427, 0.0380, 0.0369, 0.0378, 0.0345, 0.0357, 0.0375, 0.0389, 0.0329, 0.0340, 0.0345, 0.0360, 0.0327, 0.0313,
        0.0329, 0.0330],
    8: [0.3449, 0.1713, 0.0949, 0.0776, 0.0633, 0.0572, 0.0541, 0.0506, 0.0444, 0.0422, 0.0387, 0.0428, 0.0366, 0.0346,
        0.0337, 0.0334, 0.0326, 0.0308, 0.0311, 0.0325, 0.0305, 0.0307, 0.0284, 0.0292, 0.0293, 0.0291, 0.0282, 0.0288,
        0.0294, 0.0307],
    16: [0.2755, 0.1490, 0.1083, 0.0787, 0.0674, 0.0605, 0.0547, 0.0497, 0.0476, 0.0443, 0.0406, 0.0404, 0.0386, 0.0363,
         0.0377, 0.0352, 0.0333, 0.0350, 0.0330, 0.0321, 0.0319, 0.0313, 0.0325, 0.0304, 0.0295, 0.0295, 0.0317, 0.0308,
         0.0306, 0.0285]
}


# 2. 绘图函数
def plot_loss_vs_rounds(data, num_rounds):
    """
    接收损失数据并生成“损失 vs. 训练轮次”图表。
    """
    # 为每条线定义不同的样式，以便清晰区分
    styles = {
        2: {'color': 'blue', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.5},
        4: {'color': 'green', 'marker': 's', 'linestyle': '-.', 'linewidth': 1.5},
        8: {'color': 'red', 'marker': '^', 'linestyle': ':', 'linewidth': 2},
        16: {'color': 'purple', 'marker': 'd', 'linestyle': '-', 'linewidth': 2}
    }

    # 生成 x 轴的轮次数据
    rounds = np.arange(1, num_rounds + 1)

    # 创建图表
    plt.figure(figsize=(12, 7))

    # 循环遍历每个量化水平并绘制其损失曲线
    for q_level in QUANTIZATION_LEVELS:
        plt.plot(rounds, data[q_level],
                 label=f'q = {q_level}',
                 color=styles[q_level]['color'],
                 marker=styles[q_level]['marker'],
                 linestyle=styles[q_level]['linestyle'],
                 linewidth=styles[q_level]['linewidth'],
                 markersize=5)

    # 设置图表的标题、坐标轴标签等
    plt.title('Test Loss vs. Training Rounds', fontsize=18, fontweight='bold')
    plt.xlabel('Training Round', fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.legend(title='Quantization Level', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置坐标轴刻度，使其更易读
    plt.xticks(np.arange(0, num_rounds + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, num_rounds + 1)
    plt.ylim(bottom=0)

    # 优化布局，防止标签重叠
    plt.tight_layout()

    # 保存图像到文件
    output_filename = "loss_vs_rounds_chart.png"
    plt.savefig(output_filename, dpi=300)

    # 在屏幕上显示图表
    plt.show()

    print(f"图表已成功保存为: {output_filename}")


# 3. 主程序入口
if __name__ == "__main__":
    plot_loss_vs_rounds(loss_data, NUM_ROUNDS)