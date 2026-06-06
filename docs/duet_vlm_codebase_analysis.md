# DUET-VLM 代码库完整解读

> 本文档帮助你快速理解 [DUET-VLM](https://github.com/AMD-AGI/DUET-VLM) 项目中
> 每个代码文件的职责、模块间的调用关系以及数据流走向，无需逐行阅读源码。

---

## 1. 全局架构概览

DUET-VLM 的完整推理 pipeline 如下（以 LLaVA-1.5 为例）：

```
输入图像 [B, 3, 336, 336]
    │
    ▼
┌───────────────────────────────────────────────┐
│  CLIP Vision Encoder (frozen)                  │
│  clip_encoder.py → CLIPVisionTower             │
│  输出: patch features [B, 576, 1024]           │
└───────────────────────────────────────────────┘
    │
    ▼  ← Stage 1: VisionZip (视觉编码器内 token 合并)
┌───────────────────────────────────────────────┐
│  visionzip/clip_encoder_cw.py                  │
│  • 用 CLS attention 选 dominant tokens (170个)  │
│  • 对剩余 tokens 做局部聚类得 contextual (35个)  │
│  输出: compressed features [B, 206, 1024]      │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│  MM Projector (可训练的 2 层 MLP)               │
│  multimodal_projector/builder.py               │
│  1024 → 4096 (对齐到 LLaMA hidden size)        │
│  输出: [B, 206, 4096]                          │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│  序列拼接 (llava_arch.py)                       │
│  [system tokens | image tokens | question tokens]│
│  记录 image_token_posi, image_tokens 等元数据    │
└───────────────────────────────────────────────┘
    │
    ▼  ← Stage 2: PyramidDrop (LLM 内逐层 token 裁剪)
┌───────────────────────────────────────────────┐
│  Modified LLaMA Decoder (modeling_llama_pdrop.py)│
│  • 在 layer 8/16/24 后检查                      │
│  • 用下一层的 Q/K 算 text→image 注意力          │
│  • 按 ratio [1.0, 0.5, 0.25, 0.125] 逐级裁剪  │
│  输出: 压缩后的 hidden states                   │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│  LM Head → 自回归生成答案                       │
└───────────────────────────────────────────────┘
```

---

## 2. 目录结构与文件职责

### 2.1 `visionzip/` — Stage 1: 视觉编码器内 token 合并

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 1 | 暴露 `visionzip`, `visionzip_video` |
| `main.py` | 140 | **入口函数**。`visionzip(model)` 通过 monkey-patch 替换 CLIP 的 forward，`visionzip_video(model)` 同理处理 Video-LLaVA 的双塔 |
| `utils.py` | 209 | 实现 patch 后的 `CLIPAttention_forward`（额外返回 key metric）和 `CLIP_EncoderLayer_forward`（存储 metric），以及 `apply_info()` 把参数注入模型 |
| `clip_encoder_cw.py` | 153 | **核心 token 选择逻辑**。`CLIPVisionTower_VisionZip.forward()` 做两步：(1) 用 CLS attention 选 dominant tokens (2) 对剩余 tokens 用 key 相似度聚类得 contextual tokens |
| `llava_arch.py` | 260 | VisionZip 版的 `encode_images` 和 `prepare_inputs_labels_for_multimodal`，处理 anyres 多 tile 图片的空间重排 |
| `videollava_arch.py` | 338 | 同上，但处理 Video-LLaVA 的图像+视频双模态（ndim==3 是图片，ndim==4 是视频） |

**核心算法（clip_encoder_cw.py）**：
1. 拿到 CLIP 倒数第二层的 attention weights
2. CLS token 对所有 patch 的注意力求和 → 排序取 top-K 为 "dominant"
3. 剩余 patch 中，先用 attention 分数预选 `cluster_width × contextual_num` 个候选
4. 均匀选锚点，其余候选按 key 余弦相似度分配到最近锚点
5. 每个锚点的 hidden state += 被分配 token 的平均值 → 得到 "contextual" tokens
6. 输出 = [dominant tokens, contextual tokens]，总共 dominant+contextual+1(CLS) 个

### 2.2 `llava/model/` — LLaVA 模型核心

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | — | 注册模型到 HuggingFace AutoModel |
| `builder.py` | ~200 | **模型加载工厂**。`load_pretrained_model()` 处理 LoRA、量化、多种架构。`pdrop_infer=True` 时加载 PDrop 变体 |
| `llava_arch.py` | ~400 | **架构胶水层**。`LlavaMetaModel` 管理 vision tower + projector 初始化；`LlavaMetaForCausalLM` 提供 `encode_images()` 和 `prepare_inputs_labels_for_multimodal()`（标准版）/ `..._pdrop()`（DUET 版，额外记录 image token 位置元数据） |
| `multimodal_encoder/clip_encoder.py` | ~150 | CLIP 视觉塔封装：加载模型、冻结权重、`feature_select()` 选取指定层的 hidden states |
| `multimodal_projector/builder.py` | ~50 | 投影器工厂：`mlp2x_gelu`（默认 2 层 MLP）/ `linear` / `identity` |
| `language_model/llava_llama.py` | ~100 | 标准 LLaVA-LLaMA（无 token drop），作为 baseline |
| `language_model/llava_llama_pdrop.py` | ~150 | **DUET 入口**。`LlavaLlamaForCausalLM_PDrop` 的 `forward()` 调用 `pdrop_forward()` 而非标准 forward，`generate()` 同理 |
| `modeling_llama_pdrop.py` | 2185 | **Stage 2 核心实现**（下面单独详解） |

### 2.3 `modeling_llama_pdrop.py` — Stage 2: PyramidDrop 详解

这是 DUET-VLM 最核心的文件，实现了 LLM 内部的逐层视觉 token 裁剪。

**关键配置（由外部设置在 model.model 上）**：
- `layer_list = [8, 16, 24]` — 在哪些层之后做 drop
- `image_token_ratio_list = [1.0, 0.5, 0.25, 0.125]` — 每个阶段保留的比例
- `image_token_posi` — 每个样本中 image tokens 的起始位置
- `image_tokens` — 每个样本的 image token 数量

**`pdrop_forward()` 流程**：
1. 正常做 token embedding + position embedding
2. 逐层通过 decoder layer
3. 每过一层后检查：`layer_idx + 1` 是否在 `layer_list` 中？
4. 如果是（且在 prefill 阶段）→ 调用 `pdrop_rank_drop()`：
   - 取出当前 hidden states（detach，不回传梯度）
   - 用**下一层**的 LayerNorm + Q/K 投影 + RoPE
   - 选择 query tokens：
     - 训练时：answer 开始前的 token（labels 从 -100 变为有效值的位置）
     - 推理时（有 salient idx）：salient_token_finder 提供的文本显著词位置
     - 推理时（无 salient idx）：image 之后的所有文本 token
   - 算 softmax(Q_text × K_all^T / √d) → 对 attention head 取均值 → 对 query tokens 取均值
   - 得到每个 image token 的重要性分数
   - `topk(keep_length)` 保留最重要的，丢弃其余
   - 重建序列 [前缀 text | 保留的 image | 后缀 text]，更新 labels/mask
5. 如果在 decoding 阶段（seq_len==1）→ 只调整 position_ids（补偿已 drop 的偏移量）

**设计要点**：
- 用"下一层"的 Q/K 来打分，而非当前层（避免信息泄露，且下一层的表示更新）
- 训练时从 labels 自动推断 query 位置，不需要额外标注
- salient_token_finder 是可选的推理时增强（用 NLP 找出问题中的关键词作为 query）

### 2.4 `llava/salient_token_finder.py` — 文本显著词提取

**职责**：从用户问题中提取"显著词"，用于指导 Stage 2 的 token 裁剪。

**算法**：
1. 用 spaCy 做词性标注
2. 保留：疑问词（who/what/where...）、重要标点（?!:）、名词、代词、动词、形容词、副词
3. 过滤：停用词（NLTK stopwords）
4. 去重保序

**作用**：这些词对应的 text token 位置作为 `idxs` 传给 `pdrop_rank_drop()`，
成为"用哪些文本 token 去评估哪些 image token 重要"的 query。

### 2.5 `llava/aligner.py` — Token 对齐工具

**职责**：将 salient_token_finder 输出的原始词映射到 tokenizer 的 subword token 位置。

**核心函数**：`greedy_align_and_filter(rel_words, salient_tokens)`
- 处理 subword 拆分问题（如 `['O', 'CR']` 对应 `'ocr'`）
- 贪心合并连续 subword 直到匹配 salient token
- 返回 token 位置列表，直接用于 pdrop 的 query 选择

### 2.6 `llava/train/` — 训练相关

| 文件 | 职责 |
|------|------|
| `pdrop_train.py` | **主训练入口**。解析参数（包括 PDrop 的 layer_list/ratio_list 和 VisionZip 的 dominant/contextual）、构建数据集、调用 trainer。数据预处理有两个版本：普通版和 `_salient` 版（额外计算显著词索引） |
| `llava_trainer.py` | 自定义 HF Trainer：模态感知的 batch sampler（按长度分组减少 padding）、MM projector 单独学习率、DeepSpeed ZeRO-3 兼容的 checkpoint 保存 |
| `train_mem_pdrop.py` | 内存优化版训练入口（gradient checkpointing 等） |
| `pdrop_train_next.py` | LLaVA-NeXT (v1.6) 的 PDrop 训练入口 |

### 2.7 `llava/eval/` — 评测脚本

| 文件 | 职责 |
|------|------|
| `model_vqa_loader.py` | **主评测入口**。加载模型 → 应用 VisionZip → 设置 PDrop → 逐条推理 → 写 JSONL 答案 |
| `eval_textvqa.py` | TextVQA 精度计算 |
| `eval_pope.py` | POPE benchmark 精度计算 |
| `model_vqa_science.py` | ScienceQA 推理 |
| `model_vqa_mmbench.py` | MMBench 推理 |
| 其他 `eval_*.py` | 各 benchmark 的后处理/评分脚本 |

### 2.8 `llava/` 其他辅助文件

| 文件 | 职责 |
|------|------|
| `mm_utils.py` | 图像预处理工具集：resize/pad、anyres 多 tile 切分、`tokenizer_image_token()`（在 prompt 中插入 IMAGE_TOKEN_INDEX）、`KeywordsStoppingCriteria` |
| `conversation.py` | 对话模板注册表（Vicuna v0/v1、LLaMA-2、Mistral 等格式的 prompt 模板） |
| `constants.py` | 常量定义：`IMAGE_TOKEN_INDEX=-200`, `IGNORE_INDEX=-100` 等 |

### 2.9 `videollava/` — Video-LLaVA 支持

结构与 `llava/` 高度平行，关键差异：

| 差异点 | llava/ | videollava/ |
|--------|--------|-------------|
| 视觉塔 | 单塔 (CLIP) | 双塔 (LanguageBind Image + Video) |
| 输入 | 图片 [3,H,W] | 图片 [3,H,W] + 视频 [T,3,H,W] |
| VisionZip 入口 | `visionzip(model)` | `visionzip_video(model)` — 对两个塔都做 patch |
| encode | `encode_images()` | `encode_images()` + `encode_videos()` |
| PyramidDrop | 同一个 `modeling_llama_pdrop.py` | 同一个（共享 LLaMA 后端） |

核心文件 `videollava/model/language_model/llava_llama_pdrop.py` 逻辑和 llava 版本一致，
只是 `prepare_inputs_labels_for_multimodal_pdrop` 额外处理视频帧的展开。

### 2.10 `qwen2_5_vl/` — Qwen2.5-VL 独立实现

| 文件 | 行数 | 职责 |
|------|------|------|
| `modeling_qwen2_5vl_duet.py` | 2796 | **全部逻辑合一**。不用 monkey-patch，VisionZip 和 PyramidDrop 都内建在模型 forward 中 |
| `run_inference.py` | 263 | 单图推理入口 + 性能计时 |
| `eval_benchmarks.py` | 703 | 批量评测（POPE/GQA/SQA/MME/TextVQA），支持三种模式：baseline / ori_visionzip / duet |

**与 LLaVA 版的关键差异**：

| 方面 | LLaVA 版 | Qwen 版 |
|------|---------|---------|
| VisionZip 接入方式 | 外部 monkey-patch | 内建于 `forward()` |
| 注意力来源 | CLIP 倒数第二层 hook 出来 | 视觉 Transformer 最后一个 block 原生返回 attention |
| 位置编码 | 1D position_ids | 3D multimodal RoPE (temporal, height, width) |
| PyramidDrop 打分 | 用下一层 Q/K + 1D RoPE | 用下一层 Q/K + multimodal RoPE |
| 空间合并 | 无 | 2×2 PatchMerger 在 VisionZip 之前先做一次 |
| 配置方式 | 分散在多个函数/参数 | 统一 `model.configure_duet()` API |

### 2.11 `scripts/` — 训练与评测脚本

```
scripts/
├── llava/v1_5/
│   ├── pdrop_train/pretrain.sh    # LLaVA-1.5 预训练 (Stage 1 projector 对齐)
│   ├── pdrop_train/finetune.sh    # LLaVA-1.5 微调 (全参数 + PDrop)
│   └── pdrop_eval/*.sh            # 各 benchmark 评测 (textvqa/pope/gqa/...)
├── llava/v1_6/                     # LLaVA-NeXT 版本（同结构）
├── videollava/v1_5/
│   ├── finetune.sh
│   └── eval/*.sh
└── qwen/*.sh                       # Qwen 评测脚本（按 mode 参数切换 baseline/duet）
```

每个 eval 脚本的模式：设置模型路径 → 设置 PDrop 参数（layer_list、ratio_list）→
设置 VisionZip 参数（dominant、contextual、cluster_width）→ 调用 `model_vqa_loader.py`。

### 2.12 顶层文件

| 文件 | 职责 |
|------|------|
| `utils.py` (247KB) | HuggingFace `generation/utils.py` 的拷贝修改版。包含完整的 `GenerationMixin`。被复制出来是为了支持 PDrop 推理时变长序列的 generate 逻辑（标准版不支持序列长度在生成过程中变化） |
| `setup.py` | 包定义：`duet-vlm` v1.0.0，包含 llava/visionzip/videollava/qwen2_5_vl 四个子包。依赖 PyTorch 2.0+、Transformers 4.37+、accelerate、peft 等 |

---

## 3. 数据流总结（训练 vs 推理）

### 训练时数据流

```
JSON 数据 (image path + conversation)
    │
    ▼
LazySupervisedDataset (pdrop_train.py)
    ├── 加载图片 → process_images() → tensor
    ├── 格式化对话 → conversation.py 模板
    ├── tokenize → tokenizer_image_token()
    └── (可选) salient_tokens_finder → aligner → idxs
    │
    ▼
DataCollator → batch (input_ids, labels, images, idxs)
    │
    ▼
LlavaLlamaForCausalLM_PDrop.forward()
    ├── encode_images() → VisionZip 压缩 → projector 投影
    ├── prepare_inputs_labels_for_multimodal_pdrop() → 拼接序列 + 记录元数据
    └── model.pdrop_forward() → 逐层 decoder + 在指定层 rank_drop
    │
    ▼
CrossEntropyLoss (对 drop 后的序列计算)
```

### 推理时数据流

```
问题 + 图片
    │
    ▼
model_vqa_loader.py
    ├── 格式化问题 → conv 模板 → tokenize
    ├── (可选) salient_tokens_finder → idxs
    └── process_images()
    │
    ▼
model.generate()
    ├── VisionZip 压缩 (在 vision tower forward 中)
    ├── prepare_inputs_labels_for_multimodal_pdrop()
    └── pdrop_forward() (prefill: 逐层 drop; decode: 调整 position)
    │
    ▼
生成的 answer tokens → decode → 写入 JSONL
```

---

## 4. DUET-VLM 的创新点 vs 继承代码

| 组件 | 来源 | DUET 的改动 |
|------|------|------------|
| CLIP Vision Tower | LLaVA 原版 | 无改动（冻结） |
| MM Projector | LLaVA 原版 | 无改动 |
| VisionZip token 选择 | [VisionZip](https://github.com/dvlab-research/VisionZip) | 改进了 contextual token 的聚类策略（attention-based preselection + cluster_width 参数） |
| PyramidDrop 逐层裁剪 | [PyramidDrop](https://github.com/Cooperx521/PyramidDrop) | 加入了 salient token 机制（用文本语义指导视觉 token 打分） |
| 两阶段协同 | **DUET 原创** | 将 VisionZip (V2V) 和 PyramidDrop (T2V) 串联，让两者互补 |
| Salient token finder | **DUET 原创** | 用 NLP 工具提取问题关键词，作为 T2V 打分的 query |
| Qwen 统一实现 | **DUET 原创** | 把两阶段内建到 Qwen2.5-VL 架构中，提供 `configure_duet()` API |
| 训练集成 | **DUET 原创** | VisionZip + PDrop 联合训练（不只是推理加速） |

---

## 5. 关键超参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `dominant` | 170 | VisionZip 保留的高注意力 token 数 |
| `contextual` | 35 | VisionZip 聚类得到的上下文 token 数 |
| `cluster_width` | 4 | 聚类候选池大小 = cluster_width × contextual |
| `layer_list` | [8, 16, 24] | PyramidDrop 在这些层后执行 drop |
| `image_token_ratio_list` | [1.0, 0.5, 0.25, 0.125] | 每阶段保留比例（首个 1.0 自动补上） |
| `use_salient_tokens` | True/False | 是否用 NLP 提取的显著词做 query |

**token 数量变化示例**（LLaVA-1.5, 原始 576 个 patch token）：
- VisionZip 后：170 + 35 + 1(CLS) = 206 tokens
- PyramidDrop 第一次 (layer 8)：206 × 0.5 = 103
- PyramidDrop 第二次 (layer 16)：206 × 0.25 = 51
- PyramidDrop 第三次 (layer 24)：206 × 0.125 = 25
- 最终进入 LM head 的视觉 token 只有 25 个（压缩率 95.7%）

---

*文档版本: 2026-06-06*
