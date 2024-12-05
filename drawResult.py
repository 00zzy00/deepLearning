import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('scid_epoch_results_tab.csv')

# 提取所需的列数据
epoch = df['epoch']
plcc = df['plcc']
srocc = df['srocc']
loss = df['loss']



# 绘制折线图
plt.plot(epoch, plcc, label='PLCC')
plt.plot(epoch, srocc, label='SROCC')
# 添加标签和标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.xlabel("轮次")
plt.ylabel("斯皮尔曼秩序相关系数/皮尔森线性相关系数")
# plt.title('PLCC and SROCC over Epoch on SCID')
plt.legend()

plt.savefig('plcc_srocc_plot on SCID-0.png')
plt.clf()

plt.plot(epoch, loss, label='损失值')
plt.xlabel("轮次")
plt.ylabel("损失值")
# plt.title('LOSS over Epoch on SCID')
plt.legend()

# 设置纵坐标分度值为0.2
# plt.yticks([i / 100 for i in range(60, 110, 2)])  # 分度值范围为0.5到1，每0.2为一个分度

plt.savefig('loss_plot on SCID-0.png')
# 显示图形
plt.show()
