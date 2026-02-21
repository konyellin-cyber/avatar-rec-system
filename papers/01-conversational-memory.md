# 对话记忆与检索

Conversational Memory & Retrieval

---

## <a name="secom"></a>SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents

**⭐ 优先级: P0 | 相关性: ⭐⭐⭐⭐⭐ | 作者: Microsoft Research**

### 基本信息
- **发表**: ICLR 2025 (International Conference on Learning Representations)
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

#### 补充数据集（PDF Table 8-9）

| 数据集 | SECOM GPT4Score | 最佳 baseline |
|--------|----------------|--------------|
| CoQA | **98.31** | COMEDY: 97.48 |
| Persona-Chat | **78.34** | MemoChat: 76.83 |

### 5.5 人工评估（PDF Table 10）

10 名人工标注员从 5 个维度评估：**Coherence, Consistency, Engagingness, Humanness, Memorability**
- 人工评估排名与自动指标一致
- SECOM 在 Memorability（记忆相关性）维度优势最为显著

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

_最后更新: 2026-02-21_
