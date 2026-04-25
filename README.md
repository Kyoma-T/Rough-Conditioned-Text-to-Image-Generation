# 基于 ControlNet 的局部自适应控制训练与数据构建

本项目基于 ControlNet 和 Stable Diffusion1.5 预训练模型，通过在 **预训练 ControlNet** 上增加一个 **Control Scale Predictor (CSP)**，在解码阶段预测像素级的局部控制强度 `alpha`，从而实现：旨在条件图像和文本提示的冲突区域，实现更符合文本语义的生成结果。

---

## 项目亮点

| 特性 | 描述 |
|------|------|
| **粗糙条件文生图** | 可根据文本提示和文本提示适当冲突的条件图像来生成图像 |
| **数据构造** | 提供从原始图像、条件图生成、候选 alpha 图筛选的冲突条件图及提示词数据集构造管线 |
| **低样本需求** | 只需1000张训练样本，即可实现冲突条件文生图效果，并比controlnet方法效果更好 |
| **复现友好** | 提供环境文件、配置文件、数据流水线、训练入口、推理脚本和日志/恢复能力 |


---

## 项目架构
```text
.
├─ pipeline_dataset.py              # 数据构建主流程
├─ tutorial_train.py               # 训练入口
├─ tutorial_dataset.py             # 训练数据读取
├─ quick_compare_infer.py          # 单图推理对比
├─ make_canny.py                   # 推理可视化辅助脚本
├─ audit_consistency.py            # 训练一致性检查
├─ cldm/                           # ControlNet / Stable Diffusion 相关主干代码
├─ ldm/                            # Latent Diffusion 相关代码
├─ models/
│  └─ cldm_v15.yaml                # 模型结构配置
├─ data/
│  └─ get_image/
│     └─ configs/                  # 数据构建配置
└─ environment.yaml                # 推荐环境文件
```
---

## 快速开始

### 配置环境

推荐使用 Conda：

```bash
conda env create -f  environment.yaml
conda activate sc_canny
```

### 模型准备

本仓库默认不附带大模型权重。请自行准备并放到 `models/`：

- `v1-5-pruned.ckpt`
- `control_v11p_sd15_canny.pth`
- 你训练得到的 checkpoint（如果要跑 `c_ada`）

> [Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt)
> [ControlNet](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth)
> [checkpoint]()

### 数据构建

我们准备了数据集，您可以在[这里]()目录下下载。

如果您想自己构建更大的数据集，我们提供了一个数据构建流水线,也可以参考docs/data_building.md：

### 参数说明
默认配置文件位于：

- `data/get_image/configs/paths.json`
- `data/get_image/configs/inference.json`
- `data/get_image/configs/class_map.json`
- `data/get_image/configs/alpha_list.json`

#### `paths.json`

| 参数 | 说明 |
|------|------|
| `openimages_root` | 原始图片目录 | 
| `work_root` | 整个数据流水线的工作目录，conditions/generated/manifests 都会写到这里 | 
| `prompt_json_root` | 原图 prompt 标注目录，脚本会读取 `<class>.json` | 
| `smart_ckpt` |训练得到的模型权重| 

#### `inference.json`

| 参数 | 说明 | 
|------|------|
| `width` / `height` | 输入和生成分辨率 | 
| `canny_low_threshold` | Canny 下阈值，影响边缘数量 | 
| `canny_high_threshold` | Canny 上阈值 | 
| `cfg_scale` | 文本引导强度 |
| `batch_size` | `generate` 阶段每批生成数量 |
| `seed` | 基础随机种子，实际每对样本会再做稳定扰动 |
| `mode` | 推理模式，`c_fix` 表示固定控制；`c_ada` 表示启用自适应控制 |

#### `class_map.json`

 类别替换规则

| 规则 | 说明 |
|------|------|
| `key` | 原始类别 `clsinit` | 
| `value` | 候选替换类别列表 `clsalt` | 

#### `alpha_list.json`

| 参数 | 说明 | 
|------|------|
| `alphas` | 候选控制强度列表，`generate` 会对每个 alpha 生一张图 | 

### How to train

```bash
python tutorial_train.py
```

### How to test

```bash
python quick_compare_infer.py \
  --image path/to/input.png \
  --prompt "your prompt" \
  --smart-ckpt path/to/your_smartcontrol.ckpt \
  --device cuda \
  --output-dir quick_compare_outputs/demo
```

---

## 参考文献

本复现基于以下论文：

```bibtex
@article{liu2024smartcontrol,
  title={SmartControl: Enhancing ControlNet for Handling Rough Visual Conditions},
  author={Liu, Xiaoyu and Wei, Yuxiang and Liu, Ming and Lin, Xianhui and Ren, Peiran and Xie, Xuansong and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2404.06451},
  year={2024}
}
```

在论文基础上，我们进行了以下修改：
- 优化了模型结构，减少了参数量，提高了推理效率。
- 扩大了数据集规模，并提出完整的数据构建管线。
### 修正开源代码与论文差异

在对比论文和开源代码后，我重点处理了以下实现点：

- 在训练目标中加入 `L = L_LDM + lambda_c * L_c`
- 支持 `m_conflict` / `m_bg` 的监督接口
- 当显式 mask 不可用时，提供 `diff_proxy` 和论文进行对齐
- 原代码没有实现controlnet的参数冻结，导致模型训练不耦合，我们进行了修改，效果更好。

### 构建 rough-condition 训练数据闭环

论文没有给出训练数据构造脚本，我们自己构建了一个数据构建流水线，并通过扩大数据集规模，提高了模型的泛化能力。

1.下载原始图片
2. 从原始图像生成 canny 条件图  
3. 根据 `class_map` 和 `alpha_list` 生成候选图  
4. 生成 preview panel，便于人工挑选合理 alpha  
5. 记录 `selection.csv`，要求保留样本必须显式标记 `plausible=1`  
6. 将筛选结果打包成训练所需的 control/image panel 与 manifest  

### 补齐快速测试和对比性能脚本

当前版本还额外补齐测试脚本，用于对比模型的推理结果。

- `quick_compare_infer.py`：同一输入下对比 `pretrain(c_fix)` 与 `ours(c_ada)`
- `make_canny.py`：输出更完整的 4/5 列推理可视化
- `pipeline_dataset.py`：多阶段数据构建主入口

## To do