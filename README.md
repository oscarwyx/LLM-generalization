# LLM-project

从公开原料重新构建 SQuAD 1.1 生成式问答实验：下载数据 → 清洗、查重与文章划分 → Qwen3-0.6B baseline → 固定 3-shot prompting → LoRA → 本次运行报告。仓库不附带历史成绩，也不以旧预测代替新计算。

## 安装与运行

使用 **Python 3.12、Windows 或 Linux、支持所装 CUDA 版本的 NVIDIA GPU**。8 GB RTX 5060 是本项目使用的参考硬件；显存需求还取决于精度、batch 和输入长度。此实现不支持在 CPU 或 macOS 上完成训练。首次运行需联网下载原料和约 1.5 GB 模型；磁盘还需容纳数据、缓存与 adapter。

```bash
git clone https://github.com/sunweiyi2026-ship-it/LLM-project.git
cd LLM-project
conda create -n genqa python=3.12 -y
conda activate genqa
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA 12.8 安装命令适用于相应驱动和显卡；不能仅凭 `torch.cuda.is_available()` 判断兼容性。程序会先执行实际 CUDA 计算检查。

默认完整运行：

```bash
python -m genqa run --config configs/experiment.json
```

`run` 包含数据准备，并为本次预测、训练和报告创建新的 `runs/<运行ID>/`。原料与模型缓存可以复用；模型从固定底座新建 LoRA，不续训旧 adapter，不复用历史预测。

也可以先检查数据准备，再运行实验或重建某次报告：

```bash
python -m genqa prepare --config configs/experiment.json
python -m genqa run --config configs/experiment.json
python -m genqa report --run-dir runs/<运行ID>
```

先确认环境能跑通，可执行小样本烟测：

```bash
python -m genqa run --config configs/experiment.json --smoke
```

烟测使用 16/8/8 条训练、验证、测试样本，单独标记为非正式运行；不能用它的指标或消耗代表完整实验。

Notebook 复用同一套 Python 实现。安装环境后注册内核：

```bash
python -m ipykernel install --user --name genqa --display-name "Python (GenQA)"
```

在 Jupyter 中打开 [notebooks/experiment.ipynb](notebooks/experiment.ipynb)，选择 `Python (GenQA)`，重启内核并从头运行。若尚未安装 Jupyter，可执行 `python -m pip install notebook`，再执行 `python -m notebook`。保存后打开已有输出只是显示结果；重新运行才会重新计算。

## 数据与比较方法

原文、问题及答案来自 [MRQA 2019](https://github.com/mrqa/MRQA-Shared-Task-2019) 的 SQuAD 子集；使用 [官方 SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) 恢复文章身份。MRQA 下载地址中的 `release/v2` 是发布版本，**不是 SQuAD 2.0**；本任务只包含有答案问题。

`prepare` 从公开原料执行规范化查重、5-word 近重复处理和文章映射，以 data seed 42 划分训练/验证文章，并保存数据、排除统计、来源哈希和配置。训练选样还会使用固定 tokenizer 检查完整序列长度与训练 token 预算。

| 集合 | 默认用途与规模 |
|---|---|
| 训练 | 从训练池固定选出 2,048 题；每个训练 seed 完整训练 2 轮 |
| 验证 | 256 题；检查 baseline、prompting 和两个 LoRA epoch |
| 测试 | 来自公开 dev 的 45 篇文章，每篇 20 题，共 900 题；与训练/验证文章隔离 |

新测试集不依赖私人历史预测目录，也不复刻旧项目的历史排除集合。因此，它与旧项目的 900 题不应视为同一测试集，旧分数不能直接搬来比较；公开 dev 也不是隐藏测试集。

三种方法使用同一底座及同一批评测题：

- **Baseline**：零样例短答案提示，不更新权重。
- **3-shot prompting**：明确要求最短原文答案，示例仅取自本次选定的训练数据，不更新权重。
- **LoRA**：沿用 baseline 提示，从原始底座新建 adapter；固定训练 seed 17、2 轮、r=8、alpha=16、dropout=0.05。每轮独立评验证集，最终固定第 2 轮，不按测试分数选择配置。

Baseline、prompting 和最终 LoRA 都对完整训练/验证/测试集合生成答案。训练集分数只作拟合诊断，尤其 few-shot 的三个示例本就属于训练数据；泛化结论看验证/测试集。

## 训练 token 预算

默认 `training.max_train_tokens=1_000_000`，按**每个训练 seed**计量。一个 input token 指训练序列中的非 padding token，包括 prompt、参考答案和 EOS；答案监督 token 是其中一部分，不能再相加。

准备阶段先计算 `2 × Σ每题完整序列长度`。若固定的 2,048 题训练两轮超过上限，程序报错，不偷偷改成更短的题或少训练一轮。训练时还累计 FP16 溢出重试消耗的非 padding token，重试同样占用上限；超限会停止并标记未完成。报告区分计划消耗、成功完成训练的消耗，以及包含重试的实际累计消耗。

历史实验的同一口径曾为 **935,638 input tokens**，其中 **25,658 为监督 tokens**。这只是说明预算口径的历史参照，不是新数据准备后的承诺；本次数字以实际分词及日志为准。默认 micro-batch 4、梯度累积 4，完整运行应为 128 次更新/轮、共 256 次更新及 4,096 次样本呈现。

本仓库当前默认配置已完成原料重建和分词预检：2,048 题两轮的**计划 input 为 939,880 tokens，其中监督子集为 25,392 tokens**，低于 1M 上限。这是训练前计划值，不是已经完成训练的消耗记录；实际完成量及重试消耗仍以该次训练日志为准。

Baseline 和 prompting 的新增**训练** token 为 0，但推理需要真实计算。训练集生成评测、验证、测试、few-shot 输入，以及梯度 checkpoint 的额外内部计算不属于上述训练序列 token 计数。因此，1M 是明确的训练 token 上限，不代表总项目 token、FLOPs、时长或费用上限。

## 指标、结果与复现范围

EM 与 token F1 由本次自由生成答案计算：按 SQuAD/MRQA 规则规范化，多参考取最高匹配；只去除输出两端空白，不借参考答案抽取预测。统一使用 non-thinking、greedy、最多 64 个新 token、完整输入上限 8,192 token。

训练日志另记答案位置 NLL 和 teacher-forced token accuracy；后者给定正确答案前缀，不等于自由生成准确率。评测的 NLL/PPL/TF accuracy 使用 eval seed 42 固定的每集合最多 64 题；EM/F1 使用整个集合。

每次运行保留配置、原料/数据/代码/模型哈希、实际 dtype 和环境、逐题预测、训练日志与 adapter。结果只对应本次配置和数据；默认是一个训练 seed，不能解释成多 seed 均值。

固定模型 revision、数据顺序、seed 和确定性选项有助于复现，但不能保证跨 GPU、CUDA、操作系统或库版本逐位相同。`precision="auto"` 会选择原生 BF16，否则 FP16；实际精度会被记录。严格确定性遇到不支持的算子会报错，不静默降级。

本次交付已完成的验证：

- 14 项 CPU 单元测试通过，覆盖数据处理与 token 预算。
- 从缓存的官方原料独立重建两次，输出 JSONL 与 manifest 的 SHA256 完全一致。
- 在 Windows、Python 3.12、RTX 5060 上，以严格确定性设置完成 **16/8/8** 小样本 CLI 全流程：原料重建、baseline/3-shot 真实预测、LoRA 两轮训练、逐轮验证和报告均通过。

**本次未执行新数据划分下完整的 2,048×2 GPU 训练或完整评测。**正式配置已通过 939,880 input tokens 的训练预算预检；运行上面的默认 `run` 命令后，才会生成该次完整成绩。小样本流程验证不代表正式模型效果，也不代表已经测试了其他 GPU 或 Linux 环境。

完整可调配置见 [configs/experiment.json](configs/experiment.json)。数据、模型和结果路径都相对仓库解析，不需要作者机器上的文件。
