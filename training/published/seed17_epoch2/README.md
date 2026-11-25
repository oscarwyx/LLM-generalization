# 已训练 LoRA 与完整运行结果

来源运行：`20261115T182055_377596Z`。

| 方法 | Test EM (%) | Test F1 (%) |
|---|---:|---:|
| Baseline | 22.1111 | 44.5315 |
| Prompting | 24.4444 | 49.1689 |
| LoRA | 77.1111 | 86.1260 |

直接加载：在仓库根目录按主 README 安装依赖后，运行

```bash
python -m genqa pretrained --verify-only
python -m genqa pretrained --split test
```

`adapter/` 中为 epoch 2 的两个权重文件；`data/` 为本轮固定训练、验证和测试数据；`configs/` 为原提示词；`reference/` 为已完成结果和来源日志。第一次加载仍需下载固定 revision 的 Qwen3-0.6B 底座。

重新评测总是保存到新 runs/pretrained_时间 目录，不覆盖本目录，也不重新训练。原始本机路径已在元数据中改为 `<source_project>`，只用于来源说明，不参与加载。adapter 配置中已固定公开底座模型名和 revision，adapter 权重字节未改。
