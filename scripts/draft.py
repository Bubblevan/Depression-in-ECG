from graphviz import Digraph

# 创建有向图
dot = Digraph(comment='ECG Depression Detection Pipeline', format='png')
dot.attr(rankdir='LR', size='24,18', splines='false')  # 设置方向为从左到右，图片比例为4:3

# 设置全局背景颜色为黑色
dot.attr(bgcolor='black')

# 定义样式：节点和边的颜色为白色
dot.attr('node', shape='box', style='rounded', fontname='Arial', fontsize='20', color='white', fontcolor='white')
dot.attr('edge', color='white')

# 添加子图 A
with dot.subgraph(name='cluster_A') as c:
    c.attr(style='rounded', color='white', label='Data Reading', labelfontsize='24', fontcolor='white')  # 设置字体大小
    c.node('A', 'Data Reading')
    c.node('A1', '*.bin', shape='ellipse')
    c.node('A2', '*.mat', shape='ellipse')
    c.edges([('A', 'A1'), ('A', 'A2')])

# 添加子图 B
with dot.subgraph(name='cluster_B') as c:
    c.attr(style='rounded', color='white', label='Filtering', fontcolor='white')
    c.node('B', 'Filtering')
    c.node('B1', 'PLI Suppression')
    c.node('B2', 'Savitzky-Golay')
    c.edges([('B', 'B1'), ('B', 'B2')])  # 修正：使用元组表示边

# 添加子图 D
with dot.subgraph(name='cluster_D') as c:
    c.attr(style='rounded', color='white', label='Feature Extraction', fontcolor='white')
    c.node('D1', 'R-wave')
    c.node('D2', 'HR')
    c.node('D3', 'HRV')
    c.node('D4', 'QT')
    c.node('D5', 'Freq')

# 添加子图 H
with dot.subgraph(name='cluster_H') as c:
    c.attr(rankdir='TB')
    c.attr(style='rounded', color='white', label='Machine Learning Classification', fontcolor='white')
    c.node('H1', 'KNN')
    c.node('H2', 'Random Forest')
    c.node('H3', 'Adaboost')
    c.node('H4', 'Logistic Regression')
    c.node('H5', 'etc.')

# 添加其他节点
dot.node('C', 'Baseline Removal\n(Wavelet Transform)')
dot.node('E', 'Outlier Removal')

# 添加子图 FGH
with dot.subgraph(name='cluster_FGH') as c:
    c.attr(style='invis')  # 隐藏子图框
    c.attr(rankdir='TB')
    c.node('F', 'Normalization\n(MinMaxScaler)')
    c.node('G', 'Feature Matrix\nConstruction')
    c.edges([('F', 'G')])

# 添加边
dot.edges([('A1', 'B'), ('A2', 'B')])
dot.edges([('B1', 'C'), ('B2', 'C')])
dot.edges([('C', 'D1'), ('C', 'D2'), ('C', 'D3'), ('C', 'D4'), ('C', 'D5')])
dot.edges([('D1', 'E'), ('D2', 'E'), ('D3', 'E'), ('D4', 'E'), ('D5', 'E')])
dot.edges([('E', 'F')])
dot.edges([('G', 'H1'), ('G', 'H2'), ('G', 'H3'), ('G', 'H4'), ('G', 'H5')])

# 渲染并保存图像
dot.render('ecg_pipeline', view=True)