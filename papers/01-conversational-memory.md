# 对话记忆与检索

Conversational Memory & Retrieval

---

## <a name="secom"></a>SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents

**⭐ 优先级: P0 | 相关性: ⭐⭐⭐⭐⭐ | 作者: Microsoft Research**

### 基本信息
- **发表**: 2025-02-08 (arXiv: 2502.05589) / ICLR 2025 (International Conference on Learning Representations)
- **作者**: Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, et al. (Microsoft Research)
- **链接**: https://arxiv.org/abs/2502.05589
- **Microsoft 项目页**: https://www.microsoft.com/en-us/research/project/secom/
- **OpenReview**: https://openreview.net/forum?id=xKDZAW0He3

---

## 1. 核心问题

**对话 Agent 如何有效管理和检索历史信息？**

现有方法在长期对话中构建记忆库时，通常采用以下粒度：
- **Turn-level**（对话轮次级）
- **Session-level**（会话级）
- **Summarization-based**（摘要型）

但这些方法都存在**检索准确性**和**语义质量**的问题。

### 1.1 问题的正式定义（补充自 PDF Section 2.1）

论文将问题形式化为三个阶段：

1. **记忆构建 (Memory Construction)**: 从对话历史 `H = {c_i}_{i=1}^C`（C 个 Session）构建记忆库 `M`
   - Turn-level: `|M| = Σ T_i`（每个 turn 对应一个记忆单元）
   - Session-level: `|M| = C`（每个 session 对应一个记忆单元）
   - **Segment-level (Ours)**: `|M| = Σ K_i`（每个主题段落对应一个记忆单元）

2. **记忆检索 (Memory Retrieval)**: 给定用户请求 `u*` 和上下文预算 `N`，检索 N 个相关记忆单元
   ```
   {m_n ∈ M}_{n=1}^N ← f_R(u*, M, N)
   ```

3. **响应生成 (Response Generation)**: 将检索到的记忆单元按时间序排列，作为上下文生成回复
   ```
   r* = f_LLM(u*, {m_n}_{n=1}^N)
   ```

---

## 2. 两大关键发现

### 🔍 发现 1: 记忆粒度至关重要

| 粒度类型 | 问题 | 示意图标注 |
|---------|------|-----------|
| **Turn-level** | 太细粒度 → 碎片化、不完整的上下文 | 🎯 单条消息，相似度 0.42 |
| **Session-level** | 太粗粒度 → 包含太多无关信息 | 🎯 整个会话，相似度 0.38 |
| **Summary-based** | 摘要时丢失关键信息 | 信息损失 |
| **Segment-level (Ours)** | ✅ 平衡：包含相关连贯信息 + 排除无关内容 | 🎯 主题段落，相似度 0.68 |

**实验数据**:
- Segment-level 在 LOCOMO 和 Long-MT-Bench+ 基准上**显著优于**所有 baseline
- 检索召回率提升 20%-30%
- 响应质量（GPT-4 Score）提升 9.46 分

#### 📊 Chunk Size 实验（补充自 PDF Figure 2a）

论文在 Long-MT-Bench+ 上测试了不同 chunk size（1/5/10/25/50 turns/chunk）对响应质量的影响：
- 固定检索预算为 50 turns
- **最优 chunk size ≈ 5-10 turns**（对应 Segment-level）
- 过小 (1 turn) → BLEU/ROUGE 最低
- 过大 (50 turns) → 质量也下降

### 💡 发现 2: 记忆去噪显著提升检索

自然语言具有**内在冗余性**（Shannon, 1951），这些冗余对检索系统来说是噪声。

**解决方案**: 使用 **Prompt Compression** 方法（如 LLMLingua-2）在检索前对记忆进行去噪。

**效果**（补充自 PDF Figure 3）:
- **BM25 检索器**: 压缩率 50%-75% 时，Recall@K 从 ~87% 提升到 ~92%
- **MPNet 检索器**: 压缩率 50%-75% 时，Recall@K 从 ~0.984 提升到 ~0.987
- 去噪后，查询与**相关** Segment 的相似度从 ~0.283 提升到 ~0.293
- 去噪后，查询与**无关** Segment 的相似度从 ~0.280 降低到 ~0.278
- **关键**: 压缩率超过 90% 时效果开始下降，最优区间为 **50%-75%**

---

## 3. SeCom 方法

**Se**gmentation + **Com**pression = **SeCom**

### 架构设计

```
Long Conversation History
         ↓
┌────────────────────────────┐
│ 对话分段模型 (Segmentation) │
│ - 识别主题边界              │
│ - 按主题切分               │
└────────────────────────────┘
         ↓
   Topical Segments
   (s₁, s₂, ..., sₖ)
         ↓
┌────────────────────────────┐
│ 压缩去噪 (Compression)      │
│ - LLMLingua-2 去除冗余     │
└────────────────────────────┘
         ↓
   Denoised Memory Bank
         ↓
┌────────────────────────────┐
│ 检索 (Retrieval)           │
│ - 根据查询召回 Top-N Seg   │
└────────────────────────────┘
         ↓
   Retrieved Segments
         ↓
┌────────────────────────────┐
│ 生成 (Generation)          │
│ - 直接拼接 Segments        │
│ - 避免再次摘要造成信息损失  │
└────────────────────────────┘
```

---

## 4. 技术细节

### 4.1 对话分段模型 (Conversation Segmentation Model)

**目标**: 将长对话 `c` 分割为 `K` 个主题连贯的段落

**形式化**:
```
f_I(c) = {s₁, s₂, ..., sₖ}
其中 sₖ = {t_pₖ, t_pₖ₊₁, ..., t_qₖ}
约束: pₖ ≤ qₖ, pₖ₊₁ = qₖ + 1 (连续不重叠)
```

#### 方式 A: Zero-Shot 分段（PDF Section 2.2, Figure 6）

使用 GPT-4 作为分段模型，核心 Prompt 结构：
```
Goal: 将多轮对话按主题分段为语义连贯的单元
输入格式: "[Exchange (N)]: [user]: ... [agent]: ..."
输出格式: JSONL，每行包含:
  - segment_id: 段落索引
  - start_exchange_number: 起始轮次号
  - end_exchange_number: 结束轮次号
  - num_exchanges: 轮次数量

约束条件:
  ✓ 不遗漏任何轮次 (No Missing Exchanges)
  ✓ 段落之间不重叠 (No Overlapping)
  ✓ 轮次数之和 = 总轮次数 (Accurate Counting)
```

#### 方式 B: Reflection 分段（PDF Section 2.2, 有限标注数据）

当有少量标注数据时，受 prefix-tuning 和 reflection 机制启发：
1. 将分段 Prompt 视为可优化的 "prefix"
2. 通过 LLM 自反思（self-reflection）**迭代优化** Prompt
3. 从标注数据中学习 "segmentation rubric"（分段准则）
4. **仅需 100 个标注样本**即可超越在完整训练集上训练的 baseline

**三种模型的实际表现**:

| 分段模型 | LOCOMO GPT4Score | Long-MT-Bench+ GPT4Score | 资源需求 |
|---------|------------------|--------------------------|---------|
| GPT-4 Seg | 69.33 / 71.57 | 88.81 / 86.67 | 高（API调用） |
| Mistral-7B Seg | 66.37 | 86.32 | 中（7B参数） |
| RoBERTa Seg | 61.84 | 81.52 | 低（~125M参数） |

#### 完整分段模型对比（PDF Table 11, 含所有 Baseline）

| 方法 | LOCOMO GPT4Score | LOCOMO BLEU | LMB+ GPT4Score | LMB+ BLEU |
|------|------------------|-------------|----------------|-----------|
| FullHistory | 54.15 | 6.26 | 63.85 | 7.51 |
| Turn-Level | 65.58 | 7.05 | 84.91 | 12.09 |
| Session-Level | 63.16 | 7.45 | 82.31 | 10.62 |
| ConditionMem | 65.92 | 3.41 | 85.69 | 12.16 |
| MemoChat | 65.10 | 6.76 | 85.14 | 12.66 |
| COMEDY | — | — | 84.48 | 12.30 |
| **SECOM (RoBERTa-Seg)** | **61.84** | **6.41** | **81.52** | **11.27** |
| **SECOM (Mistral-7B-Seg)** | **66.37** | **6.95** | **86.32** | **12.41** |
| **SECOM (GPT-4-Seg)** | **69.33** | **7.19** | **88.81** | **13.80** |

> 所有 SECOM 变体在 Long-MT-Bench+ 上均超越对应 baseline。即使最轻量的 RoBERTa-Seg 也在 LOCOMO 上与 baseline 竞争力相当，而 Mistral-7B-Seg 已超越所有 baseline。

#### 分段模型独立评估（PDF Table 4）

在三个标准对话分段数据集上的表现：

| 数据集 | 方法 | Pₖ↓ | WD↓ | F1↑ | Score↑ |
|--------|------|------|------|------|--------|
| DialSeg711 | 最佳 baseline (CSM) | 0.278 | 0.302 | 0.610 | 0.660 |
| DialSeg711 | **Ours (zero-shot)** | **0.093** | **0.103** | **0.888** | **0.895** |
| DialSeg711 | **Ours (w/reflection)** | **0.049** | **0.054** | **0.924** | **0.936** |
| SuperDialSeg | 最佳 baseline | 0.462 | 0.467 | 0.381 | 0.458 |
| SuperDialSeg | **Ours (zero-shot)** | **0.277** | **0.289** | **0.758** | **0.738** |
| TIAGE | 最佳 baseline (CSM) | 0.400 | 0.420 | 0.427 | 0.509 |
| TIAGE | **Ours (zero-shot)** | **0.363** | **0.401** | **0.596** | **0.607** |

> 分数公式: `Score = (2*F1 + (1-Pₖ) + (1-WD)) / 4`

### 4.2 压缩去噪 (Compression-Based Denoising)

**核心思想**: 自然语言的冗余是检索噪声

**方法**: 使用 LLMLingua-2 对记忆单元进行压缩

**公式**:
```
{m₁, m₂, ..., mₙ} ← f_R(u*, f_Comp(M), N)
```
- `u*`: 用户查询
- `f_Comp`: 压缩模型（LLMLingua-2, 基于 xlm-roberta-large）
- `M`: 记忆库
- `N`: 上下文预算（Top-N）
- **压缩率**: 75%（保留原文 75% 的 token）

**消融实验（PDF Table 2）**:

| 配置 | LOCOMO GPT4Score | Long-MT-Bench+ GPT4Score |
|------|-----------------|--------------------------|
| SECOM（完整） | 69.33 | 88.81 |
| −Denoise | 59.87 (-9.46) | 87.51 (-1.30) |

- 移除去噪机制后，LOCOMO 上性能下降高达 **9.46 分**
- 证明去噪是系统成功的关键，尤其对长对话场景

---

## 5. 实验结果

### 5.1 数据集
- **LOCOMO**: 长期对话 QA 基准（平均 300 轮, 9K tokens/样本，目前最长的对话数据集）
- **Long-MT-Bench+**: 多轮长对话测试（从 MT-Bench+ 重建，5 个 Session 合并为 1 个长对话）
- **CoQA**: 基于文本段落的 QA 对话（补充实验）
- **Persona-Chat**: 基于人设的开放域对话（补充实验）

### 5.2 实验配置（PDF Section 3）

| 配置项 | 详情 |
|--------|------|
| 响应生成模型 | GPT-3.5-Turbo（主实验）/ Mistral-7B-Instruct-v0.3（鲁棒性验证） |
| 分段模型 | GPT-4（主力）/ Mistral-7B / RoBERTa (fine-tuned on SuperDialSeg) |
| 压缩工具 | LLMLingua-2, 压缩率 75%, base model = xlm-roberta-large |
| 检索模型 | MPNet (multi-qa-mpnet-base-dot-v1) + FAISS / BM25 |
| 上下文预算 | LOCOMO: 4K tokens (~5 sessions, 10 segments, 55 turns) / Long-MT-Bench+: 1K tokens |
| 评估指标 | BLEU, ROUGE-1/2/L, BERTScore, **GPT4Score** (0-100), Pairwise比较 |

### 5.3 对比方法
- **直觉方法**: Turn-level, Session-level, ZeroHistory, FullHistory
- **SOTA 方法**: SumMem, RecurSum, ConditionMem, MemoChat, COMEDY

### 5.4 主要结论（PDF Table 1 完整数据）

#### LOCOMO 数据集

| 方法 | GPT4Score | BLEU | Rouge2 | BERTScore | #Tokens |
|------|-----------|------|--------|-----------|---------|
| ZeroHistory | 24.86 | 1.94 | 3.72 | 85.83 | 0 |
| FullHistory | 54.15 | 6.26 | 12.07 | 88.06 | 13,330 |
| Turn-Level (BM25) | 65.58 | 7.05 | 13.87 | 88.44 | 3,657 |
| Session-Level (BM25) | 63.16 | 7.45 | 14.24 | 88.33 | 3,619 |
| ConditionMem | 65.92 | 3.41 | 7.86 | 87.23 | 3,563 |
| MemoChat | 65.10 | 6.76 | 12.93 | 88.13 | 1,159 |
| **SECOM (BM25, GPT4-Seg)** | **71.57** | **8.07** | **16.30** | **88.88** | 3,731 |
| **SECOM (MPNet, GPT4-Seg)** | **69.33** | 7.19 | 13.74 | 88.60 | 3,716 |

#### Long-MT-Bench+ 数据集

| 方法 | GPT4Score | BLEU | Rouge2 | BERTScore | #Tokens |
|------|-----------|------|--------|-----------|---------|
| FullHistory | 63.85 | 7.51 | 12.87 | 85.90 | 19,287 |
| Turn-Level (MPNet) | 84.91 | 12.09 | 19.08 | 86.49 | 909 |
| MemoChat | 85.14 | 12.66 | 19.01 | 87.21 | 1,615 |
| **SECOM (MPNet, GPT4-Seg)** | **88.81** | **13.80** | **19.21** | **87.72** | 820 |

**关键发现**:
- SECOM 在更少的 token 预算下（820 vs 19,287）获得了更好的效果
- 即便 Mistral-7B 有 32K 上下文窗口，FullHistory 方法仍不如 SECOM

#### Mistral-7B 作为生成器（PDF Table 3）

| 方法 | GPT4Score (Long-MT-Bench+) |
|------|---------------------------|
| FullHistory | 78.73 |
| Turn-Level (MPNet) | 85.61 |
| **SECOM (MPNet)** | **90.58** |

> 即使使用 32K 上下文的 Mistral-7B，全量历史仍不如 SECOM，说明**有效的记忆管理比长上下文更重要**。

#### 官方 LOCOMO QA Pairs（PDF Table 6, MPNet 检索器）

使用官方 LOCOMO QA pairs 进行评估，完整指标如下：

**GPT-3.5-Turbo 作为生成器**:

| 方法 | GPT4Score | BLEU | Rouge2 | RougeL | BERTScore | #Turns | #Tokens |
|------|-----------|------|--------|--------|-----------|--------|---------|
| FullHistory | 66.28 | 7.51 | 14.07 | 27.90 | 87.82 | 293 | 18,655 |
| Turn-Level | 81.52 | 11.91 | 19.59 | 34.99 | 88.64 | 55.00 | 3,026 |
| **SECOM** | **84.21** | **12.80** | **19.90** | **35.61** | 88.59 | 56.49 | 3,565 |

**Mistral-7B-v0.3 作为生成器**:

| 方法 | GPT4Score | BLEU | Rouge2 | RougeL | BERTScore | #Turns | #Tokens |
|------|-----------|------|--------|--------|-----------|--------|---------|
| Turn-Level | 78.82 | 10.09 | 16.25 | 31.75 | 87.97 | 55.00 | 3,026 |
| **SECOM** | **80.07** | **10.67** | **16.65** | **31.81** | 87.87 | 56.49 | 3,565 |

> 注: MemoChat 无法用于 Mistral-7B，因为该模型在 JSON 生成上存在失败问题。

#### 补充数据集（PDF Table 8-9, 完整指标）

**CoQA 数据集（Table 8）**:

| 方法 | GPT4Score | BLEU | Rouge1 | Rouge2 | RougeL | BERTScore | #Tokens |
|------|-----------|------|--------|--------|--------|-----------|---------|
| Sentence-Level | 95.55 | — | — | — | — | — | — |
| Session-Level | 91.58 | — | — | — | — | — | — |
| ConditionMem | 94.32 | — | — | — | — | — | — |
| MemoChat | 97.16 | — | — | — | — | — | — |
| COMEDY | 97.48 | — | — | — | — | — | — |
| **SECOM** | **98.31** | **39.57** | **50.44** | **39.51** | **48.98** | **90.37** | 1,016 |

**Persona-Chat 数据集（Table 9）**:

| 方法 | GPT4Score | BLEU | Rouge1 | Rouge2 | RougeL | BERTScore | #Turns | #Tokens |
|------|-----------|------|--------|--------|--------|-----------|--------|---------|
| Turn-Level | 69.23 | — | — | — | — | — | — | — |
| Session-Level | 67.35 | — | — | — | — | — | — | — |
| ConditionMem | 73.21 | — | — | — | — | — | — | — |
| COMEDY | 76.52 | — | — | — | — | — | — | — |
| MemoChat | 76.83 | — | — | — | — | — | — | — |
| **SECOM** | **78.34** | **7.75** | **26.01** | **11.57** | **23.98** | **87.82** | 23.48 | 702 |

### 5.5 人工评估（PDF Table 10, 完整数据）

10 名人工标注员在 **Long-MT-Bench+** 上从 5 个维度评估（0-3 分制）：

| 方法 | Coherence | Consistency | Memorability | Engagingness | Humanness | **Average** |
|------|-----------|-------------|--------------|--------------|-----------|-------------|
| Full-History | 1.55 | 1.11 | 0.43 | 0.33 | 1.85 | 1.05 |
| Sentence-Level | 1.89 | 1.20 | 1.06 | 0.78 | 2.00 | 1.39 |
| Session-Level | 1.75 | 1.25 | 0.98 | 0.80 | 1.92 | 1.34 |
| ConditionMem | 1.58 | 1.08 | 0.57 | 0.49 | 1.77 | 1.10 |
| MemoChat | 2.05 | 1.25 | 1.12 | 0.86 | 2.10 | 1.48 |
| COMEDY | 2.20 | 1.28 | 1.20 | 0.90 | 1.97 | 1.51 |
| **SECOM (Ours)** | 2.13 | **1.34** | **1.28** | **0.94** | 2.06 | **1.55** |

**关键发现**:
- SECOM 在 **Memorability** (1.28)、**Engagingness** (0.94)、**Consistency** (1.34) 维度均最优
- SECOM 平均分 **1.55** 为所有方法中最高
- COMEDY 在 Coherence (2.20) 单项最优，但其他维度不如 SECOM
- Full-History 在 Memorability (0.43) 和 Engagingness (0.33) 上极差，说明长上下文反而会降低记忆利用效率

---

## 6. 相关工作（补充自 PDF Section 4）

### 6.1 对话记忆管理

论文梳理了现有记忆管理方法的脉络：

| 方法类别 | 代表工作 | 核心思路 |
|---------|---------|---------|
| 摘要记忆 | MPC, MemoryBank, COMEDY | 将对话历史摘要为记忆记录 |
| 递归更新 | RecurSum, ConditionMem | 递归摘要更新记忆 |
| RAG 检索 | MSC (DPR), MemoChat (LLM检索) | 检索增强生成 |
| 长上下文 | FullHistory | 直接送入全部历史（LOCOMO 证明易产生幻觉） |

**LOCOMO 数据集的关键发现**: 长上下文 LLM 容易产生幻觉，纯摘要记忆因信息损失效果次优。

### 6.2 RAG 系统中的分块粒度 (Chunking Granularity)

现有 RAG 框架（LangChain, LlamaIndex）直接依赖对话的自然结构（utterance 或 turn）来划分检索单元，**缺乏对主题边界的考虑**。

### 6.3 RAG 系统中的去噪 (Denoising)

| 方法 | 问题 |
|------|------|
| 摘要去噪 (LLM summarize) | 引入延迟和计算成本 |
| 微调检索器 embedding | 限制灵活性/可扩展性、过拟合/灾难遗忘风险 |
| **LLMLingua-2 压缩（本文）** | ✅ Plug-and-play，无需微调检索器 |

---

## 7. 对本项目的启发

### 🎯 核心洞察

1. **记忆粒度 = Segment-level（主题段落）**
   - ❌ 不是 Turn-level（单条消息）
   - ❌ 不是 Session-level（整个会话）
   - ✅ 是主题驱动的语义连贯段落

2. **记忆去噪是必须的**
   - 自然语言的冗余会干扰检索
   - 压缩去噪可以显著提升召回率

3. **直接拼接 > 摘要**
   - 检索到 Segment 后，直接作为上下文
   - 避免二次摘要造成信息损失

---

### 🛠️ 实现路径

#### 第一步：对话分段

**方案 A（高质量）**:
- 使用 GPT-4 或 Mistral-7B 进行分段
- Prompt: "将以下对话按主题分段，标注每个段落的主题"

**方案 B（轻量级）**:
- 使用 RoBERTa-scale 模型
- 或者简单规则（时间窗口 + 主题关键词变化）

**适用场景**:
- 个人记忆系统（千级数据）可以用轻量方案
- 每次新增对话后，异步分段处理

#### 第二步：记忆去噪

使用 LLMLingua-2 或类似工具：
```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()
compressed_segment = compressor.compress_prompt(segment_text)
```

#### 第三步：检索与生成

```python
# 1. 向量化查询和 Segments
query_embedding = model.encode(user_query)
segment_embeddings = [model.encode(seg) for seg in compressed_segments]

# 2. 召回 Top-K
similarities = cosine_similarity(query_embedding, segment_embeddings)
top_k_indices = np.argsort(similarities)[-k:]

# 3. 直接拼接作为上下文
context = "\n\n".join([segments[i] for i in top_k_indices])

# 4. 生成响应
response = llm.generate(context + "\n\nUser: " + user_query)
```

---

### ⚠️ 适配到个人记忆系统

| 维度 | SeCom（原论文） | 个人记忆系统 |
|------|----------------|-------------|
| **数据规模** | 长对话（数千轮） | 千级日志 |
| **分段模型** | GPT-4 / Mistral | 可用轻量规则 |
| **记忆类型** | 单一对话历史 | 多类型（insights/decisions/conversations） |
| **更新频率** | 静态测试集 | 实时增量 |

**调整建议**:
1. **分段粒度**可以更粗：按日期 + 主题标签
2. **去噪**可以简化：TF-IDF 过滤低信息密度内容
3. **检索**可以多路：时间召回 + 语义召回 + 标签召回

---

## 7. 待探索问题

- [ ] 如何定义个人记忆系统中的"Segment"？
  - 是否按日期 + 主题标签？
  - 是否需要跨日期的主题聚合？

- [ ] 轻量级分段方案的效果如何？
  - 简单规则 vs 小模型 vs GPT-4
  - **PDF 证实**: RoBERTa-scale 模型在 DialSeg711 上 Score 可达 0.936（w/reflection）

- [ ] 如何处理跨 Segment 的主题关联？
  - 是否需要 Memory-to-Memory 关系图？

- [ ] 记忆去噪的替代方案？
  - LLMLingua-2 vs TF-IDF vs 其他压缩算法
  - **PDF 结论**: 最优压缩率 50%-75%，过度压缩会损失信息

- [ ] GPT-4 zero-shot 分段的局限性？
  - **PDF Figure 11 案例**: GPT-4 倾向于更细粒度的分段（over-segmentation）
  - 在短对话/口语化对话上 WindowDiff 可达 0.80（较差）
  - Reflection 机制可显著缓解此问题

---

## 9. 论文结论（PDF Section 5）

论文系统地研究了记忆粒度对 RAG 响应生成的影响，揭示了 turn-level、session-level 和摘要方法的局限性。SECOM 通过 segment-level 记忆库 + 压缩去噪克服了这些挑战。消融实验确认了分段和去噪两个模块各自的贡献。

**核心贡献总结**:
1. 系统性地研究了记忆粒度对检索增强响应生成的影响
2. 证明了 prompt compression 可作为有效的去噪机制提升检索性能
3. 提出 SECOM：segment-level 记忆构建 + 压缩去噪，在 LOCOMO 和 Long-MT-Bench+ 上全面超越 baseline
4. 对话分段模型本身在 DialSeg711、TIAGE、SuperDialSeg 上也取得 SOTA

---

## 10. 相关资源

- **论文**: https://arxiv.org/abs/2502.05589
- **本地 PDF**: `papers/pdf/2502.05589-SeCom-Conversational-Memory.pdf`
- **项目页**: https://www.microsoft.com/en-us/research/project/secom/
- **OpenReview**: https://openreview.net/forum?id=xKDZAW0He3
- **LLMLingua-2**: https://llmlingua.com/llmlingua2.html
- **检索模型**: MPNet (`multi-qa-mpnet-base-dot-v1`) + FAISS
- **分段模型 HuggingFace**: `mistralai/Mistral-7B-Instruct-v0.3`

---

## <a name="mem0"></a>Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

**⭐ 优先级: P0 | 相关性: ⭐⭐⭐⭐⭐ | 作者: Mem0.ai**

### 基本信息
- **发表**: 2025-04-29 (arXiv: 2504.19413)
- **作者**: Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav (Mem0.ai)
- **链接**: https://arxiv.org/abs/2504.19413
- **代码**: https://mem0.ai/research / https://github.com/mem0ai/mem0
- **本地 PDF**: `papers/pdf/2504.19413-Mem0-Long-Term-Memory.pdf`

---

### 1. 核心问题

**LLM 的固定上下文窗口如何在多 Session 长期对话中保持一致性？**

现有方案的问题：
- **Full-Context**: 对话增长后 token 成本和延迟指数级增长（平均 26K tokens/对话）
- **RAG (固定 Chunk)**: 固定分块忽略语义边界，不同 chunk size 效果差异巨大
- **摘要记忆** (MemoryBank, ReadAgent): 摘要过程丢失关键细节
- **MemGPT**: OS 式分层内存，但搜索开销大（p50: 0.668s）
- **OpenAI Memory**: 无法选择性检索，且在 Temporal 问题上极差（J=21.71）

### 2. Mem0 方法

#### 2.1 核心架构（PDF Figure 2）

```
新消息 (m_{t-1}, m_t)
         ↓
┌─────────────────────────────────────┐
│      上下文构建 (Context Formation)    │
│  - 对话摘要 S（异步定期刷新）          │
│  - 最近 m=10 条消息                   │
│  - 新消息对                           │
│  → 综合 Prompt P = (S, {m_{t-m},...}, m_{t-1}, m_t) │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    记忆提取 (Extraction) — LLM ϕ(P)  │
│  从新交互中提取显著事实               │
│  → 候选记忆集合 Ω = {ω₁, ω₂, ..., ωₙ} │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    记忆更新 (Update Phase)            │
│  对每个候选 ωᵢ:                       │
│  1. 检索 Top-s=10 语义相似记忆        │
│  2. LLM Tool Call 决定操作:           │
│     - ADD: 新增（无等价记忆时）        │
│     - UPDATE: 更新（补充信息时）       │
│     - DELETE: 删除（信息矛盾时）       │
│     - NOOP: 无操作（无需修改时）       │
└─────────────────────────────────────┘
         ↓
    持久化记忆库 (Vector DB)
```

**关键设计决策**:
- **双重上下文**: 全局摘要 S + 近期消息窗口，兼顾宏观主题和细粒度时间信息
- **异步摘要**: 摘要生成独立于主流程，不引入延迟
- **LLM 决策操作**: 不使用额外分类器，直接利用 LLM 推理能力通过 Tool Call 选择 ADD/UPDATE/DELETE/NOOP
- **推理引擎**: GPT-4o-mini（所有 LLM 操作）
- **Embedding**: text-embedding-small-3

#### 2.2 Mem0g: 图增强记忆（PDF Figure 3, Section 2.2）

在 Mem0 基础上引入**知识图谱**表示：

```
有向标注图 G = (V, E, L)
  - V: 实体节点（Alice, San_Francisco, ...）
  - E: 关系边（lives_in, works_at, ...）
  - L: 语义类型标签（Person, City, Event, ...）

每个节点 v ∈ V 包含:
  1. 实体类型分类（Person, Location, Event）
  2. 嵌入向量 eᵥ
  3. 创建时间戳 tᵥ

关系表示为三元组: (vₛ, r, vᵈ)
```

**图构建流程**:
1. **实体提取**: LLM 从对话文本中识别实体及类型
2. **关系提取**: LLM 提取实体间的关系三元组
3. **冲突检测与消解**: 新信息与已有图节点冲突时，通过 LLM 判断更新/替换

**Mem0g vs Mem0 的优势**:
- 捕获实体间**复杂关系结构**（多跳推理能力更强）
- 在 Temporal 问题上提升最显著：J 从 55.51 → 58.13
- Token 开销翻倍（7K → 14K），但仍远低于 Zep（600K+）

### 3. 实验设置（PDF Section 3-4）

#### 3.1 评估基准
- **LOCOMO 数据集**: 平均 26K tokens/对话，4 类问题
  - **Single-Hop**: 单跳事实检索
  - **Multi-Hop**: 多跳推理
  - **Open-Domain**: 开放域对话
  - **Temporal**: 时间相关问题

#### 3.2 评估指标
| 指标 | 说明 |
|------|------|
| **F1 Score (F₁)** | Token 级精确率和召回率的调和平均 |
| **BLEU-1 (B₁)** | Unigram 精度 |
| **LLM-as-a-Judge (J)** | GPT-4o-mini 判断回答 CORRECT/WRONG 的正确率 |

#### 3.3 对比的 6 类 Baseline

| 类别 | 具体方法 | 说明 |
|------|---------|------|
| **已有记忆系统** | LoCoMo, ReadAgent, MemoryBank, MemGPT, A-Mem | 经典记忆增强方法 |
| **开源记忆方案** | LangMem (Hot Path) | LangChain 生态的记忆模块 |
| **RAG** | 7 种 chunk size × k∈{1,2} | 128/256/512/1024/2048/4096/8192 tokens |
| **Full-Context** | 全量对话输入 | 26K tokens 上限参考 |
| **专有模型** | OpenAI Memory (ChatGPT) | GPT-4o-mini + 手动提取记忆 |
| **记忆平台** | Zep | 商业记忆管理平台 |

### 4. 主要实验结果

#### 4.1 LOCOMO 综合性能（PDF Table 1, 含标准差）

| 方法 | SingleHop J↑ | MultiHop J↑ | OpenDomain J↑ | Temporal J↑ | **Overall J↑** |
|------|-------------|-------------|---------------|-------------|----------------|
| A-Mem* | 39.79±1.41 | 18.85±0.60 | 54.05±2.90 | 49.91±2.76 | 48.38±0.15 |
| LangMem | 62.23±1.49 | 47.92±2.30 | 71.12±0.49 | 23.43±2.75 | 58.10±0.21 |
| Zep | 61.70±0.89 | 41.35±1.91 | 76.60±0.99 | 49.31±1.50 | 65.99±0.16 |
| OpenAI | 63.79±1.29 | 42.92±0.99 | 62.29±1.69 | 21.71±1.25 | 52.90±0.14 |
| **Mem0** | **67.13±0.65** | **51.15±1.15** | 72.93±0.41 | 55.51±1.94 | **66.88±0.15** |
| **Mem0g** | 65.71±1.08 | 47.19±2.01 | **75.71±0.80** | **58.13±2.23** | **68.44±0.17** |

**关键发现**:
- **Mem0 在 SingleHop 和 MultiHop 上最优**，多跳推理能力突出
- **Mem0g 在 OpenDomain 和 Temporal 上最优**，图结构捕获时序和关系
- **vs OpenAI**: Mem0 整体提升 **~26%**（LLM-as-Judge 指标）
- **OpenAI 在 Temporal 上极差** (21.71)，说明专有记忆系统在时间推理上有缺陷
- **LangMem 在 Temporal 上也极差** (23.43)

#### 4.2 RAG Baseline 详细对比（PDF Table 2-3, Figure 5）

| Chunk Size | k=1 Overall J | k=2 Overall J |
|-----------|---------------|---------------|
| 128 | 47.77±0.23 | 59.56±0.19 |
| 256 | 33.79 | 41.31 |
| 512 | 39.20 | 44.48 |
| 1024 | 43.57 | 48.42 |
| 2048 | 45.51 | 51.22 |
| 4096 | 49.30 | 55.31 |
| 8192 | 44.53±0.13 | 60.53±0.16 |
| **Mem0** | — | **66.88±0.15** |
| **Mem0g** | — | **68.44±0.17** |

> Mem0/Mem0g 持续超越**所有** RAG 配置。最佳 RAG (k=2) 峰值约 ~61% J，而 Mem0 达到 ~67%、Mem0g 达到 ~68%+，**相对提升 10-12%**。
> RAG 的最优 chunk size 取决于 k 值：k=1 时 128 最佳 (47.77)，k=2 时 8192 最佳 (60.53)。

#### 4.3 延迟分析（PDF Table 2, 完整数据）

| 方法 | Memory Tokens | 搜索 p50 | 搜索 p95 | 总延迟 p50 | 总延迟 p95 | Overall J↑ |
|------|--------------|---------|---------|-----------|-----------|------------|
| Full-Context | 26,031 | — | — | 9.870s | **17.117s** | **72.90±0.19** |
| RAG (128, k=1) | 128 | — | — | — | — | 47.77±0.23 |
| RAG (8192, k=2) | 8,192×2 | — | — | — | — | 60.53±0.16 |
| A-Mem | 2,520 | 0.668s | — | 1.410s | — | 48.38±0.15 |
| LangMem | 127 | **17.99s** | **59.82s** | — | — | 58.10±0.21 |
| Zep | 3,911 | — | — | 1.292s | — | 65.99±0.16 |
| OpenAI | 4,437 | — | — | 0.466s | 0.889s | 52.90±0.14 |
| **Mem0** | **1,764** | **0.148s** | **0.200s** | **0.708s** | **1.440s** | **66.88±0.15** |
| **Mem0g** | 3,616 | 0.476s | — | 1.091s | 2.590s | **68.44±0.17** |

**关键发现**:
- **Mem0 搜索延迟最低**: p50=0.148s，比 LangMem 快 **121 倍**
- **vs Full-Context**: Mem0 p95 延迟降低 **92%** (17.117s → 1.440s)；Mem0g p95 延迟降低 **85%** (17.117s → 2.590s)
- **Full-Context J 最高 (72.90)** 但延迟极端：p95=17.117s，不适合生产环境
- **Token 效率**: Mem0 仅需 1,764 tokens 即达到 66.88 J，而 Full-Context 需要 26,031 tokens 才达到 72.90 J
- **LangMem 的搜索瓶颈**: 仅 127 memory tokens 但搜索 p50 高达 17.99s、p95 高达 59.82s
- **Zep 的问题**: 3,911 tokens 且需要数小时异步构建才能正确检索
- **Mem0g 图构建**: 最坏情况下不到 1 分钟即可完成

#### 延迟 vs 精度权衡（PDF Figure 4）

论文 Figure 4 通过两组柱状图可视化了搜索延迟 vs J score 和总延迟 vs J score 的权衡关系：
- **Mem0 实现了最佳的精度-延迟权衡**：搜索延迟极低 (0.148s) 且 J score (66.88) 仅次于 Full-Context
- **Full-Context 是准确率上限** (J=72.9%) 但总延迟 p50=9.87s、p95=17.12s，不可接受
- **LangMem 搜索延迟极端** (17.99s/59.82s)，尽管 token 数最少
- **记忆管理方案的优势**: Mem0/Mem0g/Zep 等记忆方案在对话长度增长时性能保持一致，不像 RAG 和 Full-Context 会随对话增长而退化

#### 4.4 Token 效率（PDF Section 4.5）

| 系统 | 平均 Token 占用 | 说明 |
|------|----------------|------|
| 原始对话 | ~26K | 全量 |
| **Mem0** | **~7K** | 自然语言记忆表示 |
| **Mem0g** | **~14K** | 图节点 + 关系 |
| Zep | **600K+** | 每个节点缓存完整摘要 + 边存事实，严重冗余 |

### 5. 与 SECOM 的对比

| 维度 | Mem0 | SECOM |
|------|------|-------|
| **时间** | 2025.04 | 2025.02 (ICLR 2025) |
| **核心思路** | 动态提取事实记忆 + 图结构 | 主题分段 + 压缩去噪 |
| **记忆粒度** | 事实级（每个记忆 = 一个 salient fact） | 段落级（每个记忆 = 一个主题段落） |
| **记忆组织** | 向量库 + 知识图谱 | 向量库（扁平） |
| **记忆更新** | 动态 ADD/UPDATE/DELETE/NOOP | 静态构建（不更新） |
| **评估数据** | LOCOMO | LOCOMO + Long-MT-Bench+ |
| **评估指标** | F1, BLEU-1, LLM-as-Judge | GPT4Score, BLEU, ROUGE, BERTScore |
| **开源** | ✅ | ❌ |
| **生产就绪** | ✅ 强调工程化 | 偏学术验证 |

> ⚠️ **注意**: 两篇论文使用的评估指标不同（J vs GPT4Score），无法直接数值对比。但 Mem0 在 LOCOMO 上超越了包括 RAG-based 在内的所有方法，而 SECOM 本质上也是 RAG-based (segment-level retrieval)。

### 6. 对本项目的启发

#### 🎯 核心洞察

1. **记忆应该是事实级而非段落级**
   - Mem0 提取 salient facts，比整个 segment 更精准
   - 但可能丢失上下文（与 SECOM 的 segment 各有取舍）

2. **动态记忆管理是关键**
   - ADD/UPDATE/DELETE/NOOP 四种操作维持记忆库一致性
   - 避免冗余和矛盾

3. **图记忆对关系推理有价值**
   - 特别是 Temporal 和 Multi-Hop 类问题
   - 但 token 开销翻倍，需要权衡

4. **异步摘要 + 近期窗口的双重上下文**
   - 全局理解 + 细粒度时间信息
   - 摘要不阻塞主流程

#### 🛠️ 实现路径

```python
# Mem0 核心流程伪代码
class Mem0:
    def process_message(self, user_msg, agent_msg):
        # 1. 构建上下文
        summary = self.async_summary_cache  # 异步刷新
        recent = self.get_recent_messages(m=10)
        prompt = (summary, recent, user_msg, agent_msg)

        # 2. 提取候选记忆
        candidates = self.llm.extract_facts(prompt)  # ϕ(P) → Ω

        # 3. 逐条更新记忆库
        for fact in candidates:
            similar = self.vector_db.search(fact, top_k=10)
            action = self.llm.tool_call(fact, similar)  # ADD/UPDATE/DELETE/NOOP
            self.execute_action(action, fact)

    def answer_query(self, query):
        memories = self.vector_db.search(query, top_k=10)
        return self.llm.generate(query, memories)
```

#### 正式算法伪代码（PDF Algorithm 1, Page 21）

```
Algorithm 1: Memory Management System

Procedure UPDATEMEMORY(F, M):
  Input: 新提取的事实集合 F, 现有记忆库 M
  for each fact f ∈ F do
    (action, m_target) ← CLASSIFYOPERATION(f, M)
    if action = ADD then
      M ← M ∪ {f}                    // 新增记忆
    else if action = UPDATE then
      M ← (M \ {m_target}) ∪ {f'}    // 替换为合并后的新记忆 f'
    else if action = DELETE then
      M ← M \ {m_target}             // 删除矛盾记忆
    else
      // NOOP: 不做任何修改
    end if
  end for
  return M

Procedure CLASSIFYOPERATION(f, M):
  Input: 候选事实 f, 现有记忆库 M
  S ← top-s semantically similar memories from M   // s=10
  if no memory in S is semantically similar to f then
    return (ADD, ∅)
  else if ∃ m ∈ S that contradicts f then
    return (DELETE, m)
  else if ∃ m ∈ S that f augments/enriches then
    if information_content(f) > information_content(m) then
      return (UPDATE, m)          // 仅当新事实信息更丰富时才替换
    else
      return (NOOP, ∅)
    end if
  else
    return (NOOP, ∅)
  end if
```

> **关键细节**: UPDATE 操作包含信息内容比较——只有当新事实比已有记忆包含更丰富的信息时才执行替换，避免信息退化。

#### ⚠️ SECOM vs Mem0 融合方向

| 来自 SECOM 的思路 | 来自 Mem0 的思路 | 融合建议 |
|------------------|-----------------|---------|
| Segment-level 分段 | 事实级提取 | 先分段，再从段内提取事实 |
| 压缩去噪 | 动态 UPDATE/DELETE | 存储时去噪 + 检索时去噪 |
| 静态记忆库 | 动态 CRUD | 增量更新 + 定期整理 |
| 扁平向量检索 | 向量 + 图结构 | 混合检索（向量 + 图 + 时间） |

---

## Memory in the Age of AI Agents: A Survey

**⭐ 优先级: P2 | 相关性: ⭐⭐⭐⭐**

### 基本信息
- **发表**: 2026-01-29 (GitHub 更新)
- **类型**: Survey + Paper List
- **GitHub**: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- **Star**: 1000+ ⭐

### 核心内容
提供了 AI Agent 记忆系统的全景综述和论文清单。

### 对本项目的价值
- 📚 文献索引工具
- 🗺️ 研究方向地图
- 🔗 后续深入阅读的入口

### 建议使用方式
当遇到具体技术问题时，可以在该 Repo 中搜索相关 keywords，找到更多参考论文。

---

## 论文对分身推荐系统的参考价值

> 分身推荐系统（avatar-rec-system）的核心是**两层检索架构**：第一层为千级个人记忆的快速检索（<2s），第二层为亿级外部知识的深度检索（<30s）。以下从两篇论文中提取对该系统各环节的具体参考。

### 1. 第一层（快速记忆检索）的直接参考

#### 1.1 记忆构建: SeCom 的分段思想 → 个人记忆的组织方式

| SeCom 发现 | 分身系统映射 |
|-----------|------------|
| Segment-level 优于 Turn-level 和 Session-level | 个人日志不应按单条消息或整天划分，而应按**主题段落**组织 |
| 最优 chunk size ≈ 5-10 turns | 日志中一个讨论主题通常 5-10 条消息，天然对应 Segment |
| 过细（碎片化）和过粗（信息稀释）都会降低召回质量 | 现有按日期的 `YYYY-MM-DD.md` 是 Session-level，粒度偏粗 |

**具体建议**: 在现有 `memory/YYYY-MM-DD.md` 日志内部，增加**主题分段标记**。可用轻量规则（主题关键词变化 + 时间间隔 >30min）自动切分，无需 GPT-4。SeCom 证明即使 RoBERTa-scale 小模型也能获得竞争力的分段质量。

#### 1.2 记忆去噪: SeCom 的压缩机制 → Token 预算控制

分身系统第一层有严格的 **Token 预算 < 1000 tokens** 约束。SeCom 的核心发现直接适用：

- **自然语言冗余是检索噪声**: 日志中的寒暄、重复表述、口语化填充词会干扰语义匹配
- **最优压缩率 50%-75%**: 可在存储时对日志 Segment 进行轻量去噪（TF-IDF 过滤低信息密度句 or LLMLingua-2 压缩）
- **去噪后 BM25 召回率从 ~87% 提升到 ~92%**: 对第一层的规则 + 文本搜索路径（ripgrep）直接有帮助

**具体建议**: 四路召回的**文本搜索路径**可以在索引时对 Segment 进行去噪压缩，提升 ripgrep 的匹配精度；**标签路径**可以从去噪后的 Segment 中提取更准确的关键词标签。

#### 1.3 记忆更新: Mem0 的动态 CRUD → 记忆自动化管理

分身系统的三大核心能力之一是**"记忆的自动化规划与处理"**。Mem0 的 ADD/UPDATE/DELETE/NOOP 机制直接提供了实现路径：

| Mem0 操作 | 分身系统场景 |
|----------|------------|
| **ADD** | 用户提到新项目、新想法、新联系人 → 新增事实到 `insights.md` |
| **UPDATE** | 用户观点演变（"之前想用 React，现在决定用 Vue"）→ 更新已有记忆 |
| **DELETE** | 过时信息（已离职的公司、已完成的项目）→ 删除矛盾记忆 |
| **NOOP** | 重复提及已知信息 → 不做修改，避免冗余 |

**关键参考**: Mem0 的 Algorithm 1 中 **UPDATE 包含信息内容比较**——只有新事实信息更丰富时才替换。这避免了简单覆盖导致信息退化，对 `insights.md` 的自动维护尤为重要。

**具体建议**: 将 Mem0 的 `CLASSIFYOPERATION` 逻辑应用到分身的记忆管理 Agent 中。每次对话后，异步提取事实 → 语义匹配已有记忆 → LLM 判断操作类型 → 自动更新 `insights.md` 和标签索引。

### 2. 第二层（海量知识检索）的参考

#### 2.1 Query Understanding: Mem0 的双重上下文 → Query Tower 增强

分身系统第二层的 Stage 1 是 **Query Understanding / User Tower**。Mem0 的上下文构建策略可直接参考：

```
Mem0 上下文 = 对话摘要 S（全局理解）+ 最近 m=10 条消息（细粒度时间）+ 新消息
                                    ↓ 映射到分身系统
分身 Query = 用户画像摘要 S（USER.md）+ 最近 3 天日志（时间窗口）+ 当前查询
```

Mem0 证明这种**异步摘要 + 近期窗口**的双重上下文显著优于只用其中一种。分身系统现有的 USER.md（静态画像）+ 最近日志（动态记忆）本质上已接近这一设计，但缺少**异步摘要刷新**机制。

**具体建议**: 定期（如每周）自动将近期日志摘要更新到 USER.md 的兴趣偏好部分，作为 Query Rewrite 的全局背景。

#### 2.2 多路召回排序: 两篇论文的共同启发

| 论文发现 | 第二层多路召回的参考 |
|---------|-------------------|
| SeCom: 直接拼接 > 摘要 | 召回的知识片段应**原文拼接**作为 LLM Re-ranking 的输入，避免二次摘要丢信息 |
| Mem0: 搜索 p50=0.148s | 第一层记忆检索可以做到极低延迟，**不应成为第二层的瓶颈** |
| SeCom: BM25 + MPNet 双检索器 | 第二层的个人记忆路径可同时使用关键词（BM25/ripgrep）和语义（embedding）双路召回 |
| Mem0g: 图结构在 Temporal 和 Multi-Hop 上优势显著 | 第二层涉及跨时间、跨实体的复杂查询时，考虑引入轻量知识图谱 |

#### 2.3 精排与多目标: 推荐系统排序公式的优化

分身系统的排序公式：`score = 相关性×α + 时间×β + 重要性×γ + 连贯性×δ`

两篇论文提供了**量化标定**的参考：
- **SeCom Table 2 (延迟分析)**: 全量上下文 J=72.90 但延迟不可接受 → **精度和延迟必须联合优化**，不能只看排序质量
- **Mem0**: 1,764 tokens 达到 66.88 J（全量的 91.7%）→ **Token 预算约束下的最优解**是动态记忆，不是全量检索
- **SeCom 人工评估**: Memorability 和 Engagingness 维度最重要 → 排序公式中**重要性γ**应有更高权重

### 3. 记忆生命周期管理

分身系统的"正向飞轮"（更多数据 → 更精准画像 → 更好记忆 → 更好对话 → 用户更愿分享）需要一个**记忆生命周期管理**机制。两篇论文的融合提供了完整的方案：

```
用户对话 → SeCom 主题分段 → Mem0 事实提取 → 记忆库
                                                  ↓
                              ┌─────────────────────────────────────┐
                              │ 记忆生命周期管理（Mem0 CRUD + SeCom 去噪） │
                              │                                     │
                              │  写入: 分段 → 去噪 → 事实提取 → ADD    │
                              │  更新: 新事实 → 语义匹配 → UPDATE       │
                              │  清理: 矛盾检测 → DELETE               │
                              │  整理: 定期压缩去噪 → 标签重建          │
                              └─────────────────────────────────────┘
                                                  ↓
                              四路召回（时间 + 标签 + 文件 + 文本搜索）
                                                  ↓
                                           轻量排序 → Top-K
```

### 4. 关键技术决策建议

| 决策点 | 建议 | 论文依据 |
|--------|------|---------|
| 分段模型选择 | RoBERTa-scale 或规则 | SeCom: RoBERTa 在 DialSeg711 Score=0.936，千级数据不需要 GPT-4 |
| 是否需要去噪 | 是，存储时去噪 | SeCom: 去噪提升 LOCOMO 9.46 分，且第一层 Token 预算紧 |
| 记忆粒度 | 事实级（从 Segment 中提取） | Mem0: 1,764 tokens 即可达到 66.88 J；SeCom Segment 提供上下文 |
| 记忆更新策略 | Mem0 CRUD 四操作 | Mem0: Algorithm 1 的信息内容比较避免退化 |
| 是否需要图结构 | Phase 1 不需要，Phase 2 考虑 | Mem0g: 图结构 token 翻倍（1,764→3,616），千级数据 ROI 不高 |
| 检索方案 | 多路并行（BM25 + 语义） | SeCom: BM25 GPT4Score=71.57 > MPNet=69.33，两者互补 |
| Token 预算分配 | 第一层 <1000, 第二层宽松 | Mem0: 精选 1,764 tokens 达到全量 91.7% 的效果 |
| 延迟控制 | 第一层 <2s 完全可行 | Mem0: 搜索 p50=0.148s，总延迟 p50=0.708s |

### 5. Phase 1 原型中可直接复用的技术点

| 技术点 | 来源 | 在 Phase 1 中的具体用法 |
|--------|------|----------------------|
| 主题分段规则 | SeCom | 日志按 `时间间隔>30min OR 主题关键词变化` 切分 Segment |
| TF-IDF 去噪 | SeCom 启发 | 对 Segment 过滤低 TF-IDF 句，压缩至 75% |
| 事实提取 Prompt | Mem0 | 从 Segment 中提取 salient facts 写入 `insights.md` |
| CRUD 分类逻辑 | Mem0 Algo 1 | 新事实 → 语义匹配已有 insights → 判断 ADD/UPDATE/DELETE/NOOP |
| 排序公式标定 | 两者 | 时间衰减参考 Mem0 近期窗口 m=10；相关性参考 SeCom 去噪后的相似度提升 |

_最后更新: 2026-02-21（PDF 视觉分析补充：完整实验数据、标准差、延迟对比、算法伪代码、人工评估详情；新增论文对分身推荐系统的参考价值分析）_
