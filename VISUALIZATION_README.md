# DNNWG 可视化文档

这个文档介绍如何使用 `visualize_dnnwg.py` 脚本生成 DNNWG 框架的详细可视化图表。

## 快速开始

### 安装依赖

```bash
pip install matplotlib numpy
```

### 生成所有可视化图表

```bash
python visualize_dnnwg.py
```

这将生成4个详细的可视化图表:

1. **dnnwg_full_architecture.png** - 完整架构图 (901KB)
2. **dnnwg_training_pipeline.png** - 训练流程图 (547KB)
3. **dnnwg_data_flow.png** - 数据流和张量维度图 (580KB)
4. **dnnwg_model_details.png** - 模型详细结构图 (687KB)

## 可视化内容详解

### 1. 完整架构图 (dnnwg_full_architecture.png)

**内容概览**:
- **阶段1: VAE (左上区域)** - 变分自编码器结构
  - 输入预训练权重 → VAE编码器 → 潜在空间 → VAE解码器 → 重构权重
  - KL散度正则化
  - 重构损失计算

- **阶段2: 扩散模型 (左下区域)** - 潜在扩散模型
  - 前向扩散过程 (添加噪声)
  - UNet去噪器详细结构
  - 时间步嵌入
  - 反向去噪过程
  - 冻结的VAE编码器/解码器

- **条件编码器 (右侧区域)** - 可选的条件生成模块
  - 方案1: MLP编码器 (更快)
  - 方案2: Set Transformer编码器 (更强大)
  - CLIP对齐模块 (可选)

- **训练损失 (右下区域)**
  - VAE损失: 重构损失 + KL散度
  - 扩散损失: 噪声预测MSE
  - CLIP对齐损失 (可选)

**颜色编码**:
- 🔴 红色: VAE编码器组件
- 🔵 青色: VAE解码器组件
- 🟦 蓝色: 扩散模型组件
- 🟩 绿色: UNet去噪器
- 🟨 黄色: 条件编码器
- 🟪 紫色: 潜在表示
- 🌸 粉色: 条件嵌入

### 2. 训练流程图 (dnnwg_training_pipeline.png)

**训练步骤可视化**:

**阶段1: VAE训练**
1. 加载预训练权重数据
2. 配置VAE (base_config_kl.yaml)
3. 训练编码器和解码器
4. 计算重构损失和KL散度
5. 优化VAE参数
6. 保存检查点到 `vae_checkpoints/`

**阶段1.5: CLIP对齐 (可选虚线路径)**
1. 提取VAE编码器
2. 计算条件特征 (使用CLIP)
3. 训练CLIP对齐模型
4. 保存编码器到 `clip_models/`

**阶段2: 扩散模型训练**
1. 加载冻结的VAE编码器
2. 配置UNet去噪器
3. 编码权重到潜在空间
4. 添加噪声 (前向过程)
5. UNet预测噪声
6. 计算扩散损失
7. 优化UNet参数
8. 保存检查点到 `ldm_checkpoints/`

**采样阶段 (绿色路径)**
1. 随机噪声 z_T
2. 逐步去噪 z_T → z_0
3. VAE解码 z → 权重

**时间线**:
- 显示整个训练流程的时间顺序
- 标记关键里程碑

### 3. 数据流和张量维度图 (dnnwg_data_flow.png)

**详细的张量形状变换**:

**VAE编码-解码数据流**:
```
[B, 2864] → Reshape → [B, 1, 64, 64]
  → Conv 1→128 → [B, 128, 64, 64]
  → Down×2 + Res → [B, 256, 16, 16]
  → Quant Conv → [B, 8, 16, 16]
  → 采样 z → [B, 4, 16, 16]
  → Post Quant → [B, 4, 16, 16]
  → Up×2 + Res → [B, 128, 64, 64]
  → Conv 128→1 → [B, 1, 64, 64]
```

**扩散模型训练数据流**:
```
[B, 4, 16, 16] → 加噪声 → [B, 4, 16, 16]
  → + 条件 c → [B, 5, 16, 16]
  → UNet层1 → [B, 256, 16, 16]
  → UNet层2 → [B, 256, 8, 8]
  → UNet中间 → [B, 256, 8, 8]
  → UNet上采样 → [B, 256, 16, 16]
  → 预测 ε̂ → [B, 4, 16, 16]
```

**条件编码器数据流**:
- **MLP路径**: [N, 10, 5, D] → Flatten → Linear → [B, 1024]
- **Set Transformer路径**: [N, 10, 5, D] → Encoder → Attention → [B, 512]

**采样生成数据流**:
```
随机噪声 [B, 4, 16, 16]
  → UNet去噪 (迭代)
  → z₀ [B, 4, 16, 16]
  → VAE解码 → [B, 2864]
```

### 4. 模型详细结构图 (dnnwg_model_details.png)

包含6个子图,展示核心模块的内部结构:

**子图1: VAE编码器结构**
- 详细的层级结构
- Input → Conv2d → ResBlock×2 → Downsample → ... → Quant Conv
- 显示每层的通道数和分辨率变化

**子图2: VAE解码器结构**
- 对称的解码器结构
- Latent z → Post Quant Conv → ResBlock×2 → Upsample → ... → Output
- 包含注意力层

**子图3: UNet去噪器结构**
- U形架构可视化
- 下采样路径 → 中间块 → 上采样路径
- Skip连接 (橙色虚线)
- 时间步嵌入注入

**子图4: ResNet Block结构**
- 主路径: GroupNorm → Swish → Conv → GroupNorm → Dropout → Conv
- Skip连接 (橙色虚线)
- 时间嵌入注入 (蓝色虚线)
- 残差相加

**子图5: Self-Attention机制**
- QKV投影
- 注意力计算: Attention = softmax(QK^T/√d)
- 加权求和: Output = Attention × V
- 残差连接
- 输出投影

**子图6: 扩散过程详解**
- **前向扩散**: x₀ → x₁ → x₂ → ... → x_T (逐渐添加噪声)
- **反向去噪**: x_T → x_{T-1} → ... → x₁ → x₀ (UNet逐步去噪)
- 数学公式:
  - 前向: q(x_t|x_{t-1}) = 𝒩(x_t; √(1-β_t)x_{t-1}, β_t I)
  - 反向: p_θ(x_{t-1}|x_t) = 𝒩(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
- 训练目标: 𝓛 = 𝔼[||ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t)ε, t)||²]

## 自定义可视化

### 单独生成某个图表

编辑 `visualize_dnnwg.py` 的 `main()` 函数,注释掉不需要的可视化:

```python
def main():
    visualizer = DNNWGVisualizer()

    # 只生成完整架构图
    visualizer.visualize_full_architecture('my_architecture.png')

    # # 不生成其他图表
    # visualizer.visualize_training_pipeline('dnnwg_training_pipeline.png')
    # visualizer.visualize_data_flow('dnnwg_data_flow.png')
    # visualizer.visualize_model_details('dnnwg_model_details.png')
```

### 修改颜色方案

在 `DNNWGVisualizer.__init__()` 中修改 `self.colors` 字典:

```python
self.colors = {
    'vae_encoder': '#YOUR_COLOR',  # 使用十六进制颜色代码
    'vae_decoder': '#YOUR_COLOR',
    # ...
}
```

### 调整图表大小

修改各个 `visualize_*` 方法中的 `figsize` 参数:

```python
fig, ax = plt.subplots(figsize=(宽度, 高度))  # 单位: 英寸
```

### 修改输出分辨率

修改 `plt.savefig()` 的 `dpi` 参数:

```python
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 默认300 DPI
# dpi=150  # 较低质量,文件更小
# dpi=600  # 高质量,适合打印
```

## 程序化使用

也可以在Python脚本中导入并使用:

```python
from visualize_dnnwg import DNNWGVisualizer

# 创建可视化器
viz = DNNWGVisualizer()

# 生成特定图表
viz.visualize_full_architecture('custom_path/architecture.png')

# 自定义颜色
viz.colors['vae_encoder'] = '#FF0000'
viz.visualize_training_pipeline('custom_pipeline.png')
```

## 技术细节

### 使用的库
- **matplotlib**: 绘图库
- **numpy**: 数值计算
- **matplotlib.patches**: 绘制形状和箭头

### 核心绘图函数

1. **`_create_box()`**: 创建带文本的圆角矩形框
2. **`_create_arrow()`**: 创建箭头,支持虚线/实线、颜色自定义
3. **`_create_data_node()`**: 创建数据节点,显示名称和张量形状

### 坐标系统

所有图表使用自定义坐标系统:
- X轴: 0-20 (部分图表)
- Y轴: 0-14 (部分图表)
- 坐标单位是任意的,用于布局

## 常见问题

### Q: 中文显示为方框?
A: 需要安装中文字体。可以修改 `plt.rcParams['font.sans-serif']` 为系统中已有的中文字体。

### Q: 图表太大/太小?
A: 修改 `figsize` 参数和 `dpi` 参数调整。

### Q: 如何导出为PDF?
A: 将 `.png` 改为 `.pdf`:
```python
visualizer.visualize_full_architecture('architecture.pdf')
```

### Q: 如何添加自己的模块?
A: 参考现有的绘图函数,使用 `_create_box()` 和 `_create_arrow()` 添加新组件。

## 图例说明

### 线条样式
- **实线箭头**: 数据流/前向传播
- **虚线箭头**: Skip连接/条件注入/损失反馈
- **颜色箭头**:
  - 黑色: 标准数据流
  - 红色: 损失反馈
  - 绿色: 生成/采样
  - 蓝色: 时间嵌入
  - 橙色: Skip连接/条件

### 框样式
- **圆角矩形**: 标准模块
- **虚线框**: 可选模块
- **背景色块**: 逻辑分组

## 示例输出

生成的图表适用于:
- 📊 论文插图
- 📖 技术文档
- 🎓 教学演示
- 💼 技术报告
- 📱 演讲PPT

## 许可

该可视化工具是DNNWG项目的一部分,遵循相同的许可协议。

## 贡献

欢迎提交改进建议和PR!

- 添加新的可视化视图
- 改进现有图表
- 优化代码结构
- 修复bug
