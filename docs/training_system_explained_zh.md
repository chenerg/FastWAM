# FastWAM 训练系统详解（训练框架 / 精度 / dtype / Dataset）

> 面向仓库当前实现（`/workspace/FastWAM`）的代码级说明，重点回答：
> 1) 训练到底用 `accelerate` 还是 `torchrun`；
> 2) precision/mixed precision 如何配置和生效；
> 3) 网络内部 dtype 流转；
> 4) dataset 如何切分、预处理、载入与对齐。

---

## 1. 训练入口与分布式框架：到底是 Accelerate 还是 Torchrun？

### 1.1 主训练框架：**Accelerate + DeepSpeed**

训练主入口是：

- `scripts/train_zero1.sh`
- `scripts/train_zero2.sh`
- 最终调用 `scripts/train.py`

两套脚本都使用 `accelerate launch` 启动，区别仅在于加载不同的 accelerate 配置（ZeRO1 vs ZeRO2）：

- `train_zero1.sh` → `scripts/accelerate_configs/accelerate_zero1_ds.yaml`
- `train_zero2.sh` → `scripts/accelerate_configs/accelerate_zero2_ds.yaml`

这两个 accelerate 配置都显式设置了：

- `distributed_type: DEEPSPEED`
- `deepspeed_config_file` 指向 `scripts/ds_configs/*.json`

因此，**训练本体并不是原生 `torchrun` + DDP 手写逻辑，而是 Accelerate 封装下的 DeepSpeed 训练栈**。

### 1.2 `torchrun` 在这个仓库里的角色

`torchrun` 在 README 中出现，但用于**文本 embedding 预计算脚本**（`scripts/precompute_text_embeds.py`）的多卡并行，不是主训练 loop：

- 单卡：`python scripts/precompute_text_embeds.py ...`
- 多卡：`torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py ...`

该脚本内部通过 `WORLD_SIZE/LOCAL_RANK` 检测分布式，并执行 `dist.init_process_group(...)`，与主训练路径（Accelerate）是分开的。

### 1.3 训练内部并行抽象

`Wan22Trainer` 中核心对象是：

- `Accelerator(...)`
- `accelerator.prepare(model, optimizer, loader, scheduler)`
- `accelerator.autocast()`
- `accelerator.backward(loss)`
- `accelerator.save_state/load_state`

说明并行、AMP、状态保存恢复都交给 Accelerate/DeepSpeed 管理。


### 1.4 `cfg.model` 是在哪里设置的？具体是什么值？

这个问题的关键是 Hydra 的配置合成顺序：

1. `scripts/train.py` 用 `@hydra.main(config_path="../configs", config_name="train")` 指定基础配置入口是 `configs/train.yaml`。
2. `configs/train.yaml` 里把 `model` 默认设成 `null`（`defaults` 中是 `- model: null`），所以仅靠基础配置时 `cfg.model` 还没有具体模型结构。
3. 真正训练时通过命令行传 `task=...`（例如 `task=libero_uncond_2cam224_1e-4`），Hydra 会加载 `configs/task/*.yaml`。
4. task 配置里用 `override /model: fastwam`（或 `fastwam_joint` / `fastwam_idm`）把模型组覆盖掉。
5. 对应模型组文件在 `configs/model/*.yaml`，其中包含 `_target_`（比如 `fastwam.runtime.create_fastwam`）以及完整的模型参数；最终这整棵配置树就是 `cfg.model`。

所以：

- 当你跑 `task=libero_uncond_2cam224_1e-4` 时，`cfg.model` 来自 `configs/model/fastwam.yaml`。
- 当你跑 `task=libero_joint_2cam224_1e-4` 时，`cfg.model` 来自 `configs/model/fastwam_joint.yaml`。
- 当你跑 `task=libero_idm_2cam224_1e-4` 时，`cfg.model` 来自 `configs/model/fastwam_idm.yaml`。

最后在 `run_training` 中，代码用 `instantiate(cfg.model, model_dtype=model_dtype, device=model_device)` 实例化该模型配置。



## 2. Precision（混合精度）配置如何生效

## 2.1 配置来源

全局训练配置在 `configs/train.yaml`：

- `mixed_precision: "bf16"`（默认）
- 注释允许值：`[no, fp16, bf16]`

任务配置一般不覆盖该项，因此默认走 bf16。

## 2.2 代码中的校验与映射

在 `src/fastwam/runtime.py` 中：

1. `_normalize_mixed_precision` 校验字符串必须是 `no/fp16/bf16`。
2. `_mixed_precision_to_model_dtype` 做映射：
   - `no` → `torch.float32`
   - `fp16` → `torch.float16`
   - `bf16` → `torch.bfloat16`
3. `run_training` 里将该 dtype 作为 `model_dtype` 传入 `instantiate(cfg.model, model_dtype=...)`。

即：**配置同时影响模型构建 dtype**。

## 2.3 与 Accelerate 的 AMP 协同

`Wan22Trainer.__init__` 中再次读取同一个配置，并传给：

```python
Accelerator(mixed_precision=self.mixed_precision, ...)
```

训练/评估 forward 使用：

```python
with self.accelerator.autocast():
    loss, loss_dict = train_model.training_loss(sample)
```

所以实际精度由两层共同决定：

- **模型参数/输入张量显式 cast 到 model_dtype**
- **前向计算由 Accelerate autocast 接管 AMP 细节**

## 2.4 DeepSpeed 配置与精度关系

`accelerate_zero1_ds.yaml` / `accelerate_zero2_ds.yaml` 里把 `mixed_precision` 设为 `null`，注释写明“precision 由 Trainer 的 Accelerator(mixed_precision=...) 与 DS JSON 决定”。

而 `ds_zero1_config.json` / `ds_zero2_config.json` 当前主要定义 ZeRO stage、bucket 等，不额外固定 fp16/bf16 块。

可理解为：**主导开关在训练 config (`mixed_precision`) + Accelerate**。

---

## 3. 网络内部 dtype 流转（从数据到 loss）

以下以 FastWAM 主模型 (`src/fastwam/models/wan22/fastwam.py`) 为主说明。

## 3.1 模型初始化 dtype

`FastWAM.from_wan22_pretrained(..., torch_dtype=...)` 会把同一个 `torch_dtype` 传给：

- Wan2.2 video expert
- ActionDiT
- VAE / text encoder（若加载）
- `proprio_encoder`（`nn.Linear(...).to(torch_dtype)`）

并记录到 `self.torch_dtype`。

## 3.2 训练输入 cast 规则

在 `build_inputs`：

- `video` → `input_video.to(device, dtype=self.torch_dtype)`
- `context` → `dtype=self.torch_dtype`
- `action` → `dtype=self.torch_dtype`
- `proprio`（若启用）→ `dtype=self.torch_dtype`
- `context_mask / action_is_pad / image_is_pad` → `torch.bool`

即：**数值特征统一进 model dtype，mask 统一 bool**。

## 3.3 loss 处的数值稳定策略

`training_loss` 里对 MSE 损失使用 `pred.float()` / `target.float()` 再计算，随后再做加权平均。

这意味着即便前向是 bf16/fp16，损失计算关键路径会提升到 fp32，减少数值误差。

## 3.4 推理/评估阶段 dtype

- 评估训练 loss 仍在 `accelerator.autocast()` 下执行。
- VAE 对 GT 视频重建时显式 cast 到 `model.torch_dtype`。
- 最终导出 action 到 CPU 时转成 `torch.float32`（便于后处理/指标）。

## 3.5 文本 embedding cache 的 dtype

`precompute_text_embeds.py` 中文本编码器使用 `torch_dtype=torch.bfloat16`，并把缓存写为：

- `context`: bf16
- `mask`: bool

后续 `RobotVideoDataset` 读取该缓存，供训练直接使用。

---

## 4. Dataset 处理与载入全链路

## 4.1 数据配置层（Hydra）

以 `configs/data/libero_2cam.yaml` / `configs/data/robotwin.yaml` 为例，定义：

- `dataset_dirs`：一个或多个 LeRobot 数据目录
- `shape_meta`：image/action/state 的 raw_shape 与目标 shape
- `num_frames`（默认 33）
- `action_video_freq_ratio`（常用 4）
- `video_size`
- `text_embedding_cache_dir`
- `context_len`
- `processor`（FastWAMProcessor）

`robotwin.yaml` 还定义了 `val` 数据集（`is_training_set: false`）。

## 4.2 多数据集聚合与切分（BaseLerobotDataset）

`RobotVideoDataset` 内部首先构造 `BaseLerobotDataset`：

- 基于 `MultiLeRobotDataset` 聚合多个 dataset_dir
- 读取各数据集 fps，要求一致
- 生成 `delta_timestamps` 用于对齐 obs/action 时间窗口
- 按 episode 做 train/val 切分（随机打乱后按比例切）
- `__getitem__` 失败时会随机重试（最多 5 次）

## 4.3 Processor 预处理（FastWAMProcessor）

`BaseLerobotDataset.__getitem__` 取到原始样本后，若设置了 processor，会调用 `processor.preprocess`：

- 图像变换链（train/val transforms）
- action/state 的结构化变换（`action_state_transforms`）
- action/state 归一化（`LinearNormalizer`）
- action/state 合并（`ConcatLeftAlign`）
- 指令文本增强（高/低层指令拼接逻辑）

## 4.4 归一化统计（dataset_stats）

`RobotVideoDataset.__init__` 行为：

- 如果训练集且 `pretrained_norm_stats` 未提供：
  - 主进程在线计算 stats
  - 广播给其他进程
  - 保存到当前 run 的 `dataset_stats.json`
- 如果提供了 `pretrained_norm_stats`：
  - 从该 json 加载并设置 normalizer

这也是 README 要求“首次任务先设 null，跑一次拿到 stats，再回填路径”的原因。

## 4.5 RobotVideoDataset 的最终样本构造

`RobotVideoDataset._get` 在 processor 之后继续整理训练所需字段：

1. **视频时间重采样**：按 `video_sample_indices` 从原始 T 中抽帧（由 `action_video_freq_ratio` 控制）。
2. **多相机拼接**：
   - LIBERO 常用 horizontal 拼接两路相机。
   - RoboTwin 可用 `robotwin` 特定布局（top + 双腕）。
3. **空间预处理**：
   - `ResizeSmallestSideAspectPreserving`
   - `CenterCrop`
   - `Normalize(mean=0.5, std=0.5)`，输出到 `[-1, 1]`
4. **维度变换**：`[T,C,H,W] -> [C,T,H,W]`
5. **时序对齐**：`proprio` 截成 `[:-1]` 与 action 对齐。
6. **文本条件**：
   - 用 `DEFAULT_PROMPT` 包装任务指令
   - 从 `text_embedding_cache_dir` 读取 `context/mask`

最终返回字段包括：

- `video`, `action`, `proprio`, `prompt`
- `context`, `context_mask`
- `image_is_pad`, `action_is_pad`, `proprio_is_pad`

## 4.6 DataLoader 与采样

Trainer 里 DataLoader 使用：

- `shuffle=False`
- 自定义 `ResumableEpochSampler`（支持断点恢复 epoch 内位置）
- `pin_memory=torch.cuda.is_available()`

同时会在训练前做一次跨 rank 的 dataset length 一致性检查，避免多进程数据长度不一致导致死锁。

---

## 5. 一句话总结（回答你的四个关键问题）

1. **训练框架**：主训练是 **Accelerate + DeepSpeed (ZeRO1/ZeRO2)**，不是直接 torchrun 训练。`torchrun` 只用于可选的文本 embedding 预计算多卡。  
2. **precision 设置**：由 `configs/train.yaml` 的 `mixed_precision`（默认 bf16）驱动；运行时映射为 `model_dtype`，并传给 `Accelerator(mixed_precision=...)` + `autocast()`。  
3. **网络内部 dtype**：输入数值张量统一 cast 到 `self.torch_dtype`（常见 bf16），mask 为 bool；loss 计算关键处转 fp32 提升稳定性。  
4. **dataset 处理载入**：LeRobot 多数据集合并 + episode 切分 + processor 归一化/变换 + RobotVideoDataset 的抽帧/拼接/resize/crop/normalize + 文本缓存读取，最终打包为训练 sample。

---

## 6. 你可以直接复核的关键文件

- 训练入口与配置：
  - `scripts/train_zero1.sh`
  - `scripts/train_zero2.sh`
  - `scripts/accelerate_configs/accelerate_zero1_ds.yaml`
  - `scripts/accelerate_configs/accelerate_zero2_ds.yaml`
  - `configs/train.yaml`
  - `src/fastwam/trainer.py`
  - `src/fastwam/runtime.py`

- 模型 dtype 与训练 loss：
  - `src/fastwam/models/wan22/fastwam.py`
  - `src/fastwam/models/wan22/wan22.py`

- 数据处理链路：
  - `configs/data/libero_2cam.yaml`
  - `configs/data/robotwin.yaml`
  - `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
  - `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
  - `src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - `src/fastwam/datasets/dataset_utils.py`

- 文本缓存预计算：
  - `scripts/precompute_text_embeds.py`
  - `README_zh.md`（训练章节）
