# DNNWG 可视化资源汇总

本项目为DNNWG框架提供了完整的可视化解决方案。

## 📁 文件清单

### 核心可视化文件

| 文件名 | 类型 | 大小 | 说明 |
|--------|------|------|------|
| `DNNWG_VISUALIZATION.md` | Mermaid | ~30KB | **主要可视化文档(推荐)** - 包含12个专业Mermaid图表 |
| `visualize_dnnwg.py` | Python | ~35KB | Matplotlib可视化脚本 |
| `可视化使用指南.md` | 文档 | ~8KB | 完整使用指南 |
| `VISUALIZATION_README.md` | 文档 | ~10KB | Matplotlib方案详细说明 |
| `CLAUDE.md` | 文档 | ~15KB | DNNWG框架完整技术文档 |

### 生成的图片 (Matplotlib)

| 文件名 | 大小 | 分辨率 | 内容 |
|--------|------|--------|------|
| `dnnwg_full_architecture.png` | 901KB | 6000×4200px | 完整架构图 |
| `dnnwg_training_pipeline.png` | 547KB | 4800×3000px | 训练流程图 |
| `dnnwg_data_flow.png` | 580KB | 5400×3600px | 数据流和张量维度 |
| `dnnwg_model_details.png` | 687KB | 6000×3600px | 模型详细结构(6个子图) |

## 🎯 快速导航

### 我想...

#### 📖 **快速了解DNNWG架构**
→ 打开 `DNNWG_VISUALIZATION.md` 查看**图表1: 完整系统架构图**

#### 🔬 **深入理解VAE**
→ 查看**图表2: VAE详细架构**和**图表5: VAE数据流**

#### 🌊 **学习扩散模型**
→ 查看**图表3: 扩散模型详细架构**和**图表11: 扩散过程详解**

#### 🚀 **开始训练模型**
→ 查看**图表4: 训练流程图**,并参考 `CLAUDE.md` 的配置说明

#### 🛠️ **实现UNet**
→ 查看**图表8: UNet去噪器详细结构**和**图表9: ResNet Block**

#### 📊 **用于论文/PPT**
→ 使用Mermaid CLI从 `DNNWG_VISUALIZATION.md` 导出高质量SVG/PNG

#### ✏️ **自定义图表**
→ 编辑 `DNNWG_VISUALIZATION.md` (Mermaid) 或修改 `visualize_dnnwg.py` (Python)

## 📊 图表索引

### Mermaid图表 (在 `DNNWG_VISUALIZATION.md` 中)

| # | 图表名称 | 关键概念 | 适用场景 |
|---|---------|---------|---------|
| 1 | 完整系统架构图 | VAE + 扩散模型 + 条件编码器 | 论文综述、项目介绍 |
| 2 | VAE详细架构 | 编码器-潜在空间-解码器 | 技术实现、代码开发 |
| 3 | 扩散模型详细架构 | 前向扩散 + UNet + 反向去噪 | 算法理解、模型训练 |
| 4 | 训练流程图 | 3阶段训练步骤 | 实验复现、教学 |
| 5 | VAE数据流 | 张量维度变化 (2864→4×16×16→2864) | 调试、内存优化 |
| 6 | 扩散模型数据流 | UNet内部数据处理 | 性能分析 |
| 7 | 条件编码器架构 | MLP vs Set Transformer + CLIP对齐 | 条件生成研究 |
| 8 | UNet详细结构 | U形网络 + Skip连接 + 时间嵌入 | 网络设计 |
| 9 | ResNet Block结构 | 残差连接 + 时间注入 | 模块实现 |
| 10 | Self-Attention机制 | QKV投影 + 多头注意力 | 注意力分析 |
| 11 | 扩散过程详解 | DDPM/DDIM + 数学公式 | 理论学习 |
| 12 | 端到端流程 | 时序交互图 | 系统演示 |

### Matplotlib图表 (已生成PNG)

| 文件 | 主要内容 | 特点 |
|-----|---------|------|
| `dnnwg_full_architecture.png` | 三阶段架构 + 连接关系 | 彩色分区,一目了然 |
| `dnnwg_training_pipeline.png` | 时间线 + 决策分支 | 流程清晰 |
| `dnnwg_data_flow.png` | 三条并行数据流 | 张量形状完整 |
| `dnnwg_model_details.png` | 6个子图细节 | 模块化展示 |

## 🎨 可视化方案对比

| 特性 | Mermaid方案 | Matplotlib方案 |
|-----|------------|---------------|
| **格式** | 纯文本Markdown | Python脚本 |
| **查看方式** | GitHub/Typora/VSCode | 生成PNG图片 |
| **编辑难度** | ⭐⭐ 简单 | ⭐⭐⭐ 需编程 |
| **图表质量** | ⭐⭐⭐⭐⭐ 专业 | ⭐⭐⭐⭐ 良好 |
| **可维护性** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐ 一般 |
| **版本控制** | ⭐⭐⭐⭐⭐ 友好 | ⭐⭐ 不友好(二进制) |
| **导出格式** | PNG/SVG/PDF | PNG |
| **交互性** | ⭐⭐⭐⭐ 部分支持 | ⭐ 静态 |
| **自定义性** | ⭐⭐⭐⭐ 样式丰富 | ⭐⭐⭐⭐⭐ 完全控制 |
| **推荐场景** | 文档、协作、论文 | 快速预览、自定义 |

**推荐**: 优先使用Mermaid方案,需要高度自定义时使用Matplotlib方案。

## 🚀 使用工作流

### 工作流1: 论文写作

```
1. 打开 DNNWG_VISUALIZATION.md
2. 选择需要的图表
3. 使用Mermaid CLI导出SVG
   mmdc -i DNNWG_VISUALIZATION.md -o figures/ -e svg
4. 在LaTeX中引用
   \includegraphics{figures/diagram_1.svg}
```

### 工作流2: 技术演讲

```
1. 查看 dnnwg_full_architecture.png (总览)
2. 查看 dnnwg_training_pipeline.png (流程)
3. 根据需要从Mermaid导出特定模块图表
4. 插入PPT,添加动画效果
```

### 工作流3: 代码开发

```
1. 阅读 CLAUDE.md (整体理解)
2. 参考图表5-6 (数据流和维度)
3. 参考图表8-10 (模块实现)
4. 在代码注释中引用图表编号
```

### 工作流4: 学习研究

```
1. 从图表1开始 (全局架构)
2. 深入图表2-3 (VAE和扩散模型)
3. 学习图表11 (数学原理)
4. 实践图表4 (训练流程)
5. 分析图表12 (端到端交互)
```

## 🔧 工具链

### 必需工具

- **查看Mermaid**: GitHub / Typora / VSCode / Obsidian
- **生成Matplotlib图**: Python 3.7+, matplotlib, numpy

### 推荐工具

- **Mermaid CLI**: 导出高质量图片
  ```bash
  npm install -g @mermaid-js/mermaid-cli
  ```

- **Typora**: 最佳Markdown编辑器
  https://typora.io/

- **VSCode插件**: Markdown Preview Mermaid Support
  https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid

- **在线编辑器**: Mermaid Live
  https://mermaid.live/

## 📖 相关文档

| 文档 | 内容 | 何时阅读 |
|-----|------|---------|
| `CLAUDE.md` | DNNWG框架完整说明 | 开始使用框架前 |
| `可视化使用指南.md` | 可视化使用方法 | 需要生成图表时 |
| `VISUALIZATION_README.md` | Matplotlib详细说明 | 自定义Python图表时 |
| `DNNWG_VISUALIZATION.md` | Mermaid图表集合 | 理解架构/制作图表时 |
| `README.md` | 项目总览 | 项目介绍 |

## 💡 最佳实践

### ✅ 推荐做法

1. **版本控制**: 使用Mermaid纯文本格式,便于Git跟踪
2. **模块化**: 复杂图表拆分为多个子图
3. **命名规范**: 使用描述性节点名称
4. **注释完整**: 添加说明文字和公式
5. **定期更新**: 代码更新后同步更新图表

### ❌ 避免做法

1. ~~将Matplotlib PNG图片提交到Git~~ → 使用Mermaid
2. ~~复制粘贴相似图表~~ → 使用样式类复用
3. ~~图表缺少标注~~ → 添加维度、参数说明
4. ~~使用过小的字体~~ → 确保可读性
5. ~~图表与代码不一致~~ → 保持同步

## 🎓 学习路径

### 初学者
```
Day 1: 阅读 CLAUDE.md + 查看图表1
Day 2: 理解图表2-3 (VAE和扩散)
Day 3: 跟随图表4训练第一个模型
```

### 进阶者
```
Week 1: 深入图表5-6 (数据流分析)
Week 2: 实现图表8-10 (自定义模块)
Week 3: 研究图表11 (算法优化)
```

### 研究者
```
研究方向1: 改进VAE → 参考图表2、5
研究方向2: 优化扩散过程 → 参考图表3、11
研究方向3: 条件生成 → 参考图表7
```

## 📞 支持

- **问题反馈**: GitHub Issues
- **功能建议**: Pull Request
- **技术讨论**: 参考 CLAUDE.md

## 📝 版本历史

- **v1.0** (2025-11): 初始版本
  - 12个Mermaid图表
  - 4个Matplotlib图表
  - 完整文档

---

**推荐起点**: 打开 `DNNWG_VISUALIZATION.md` 查看图表1 🚀
