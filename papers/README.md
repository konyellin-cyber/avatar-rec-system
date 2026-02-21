# 相关论文清单

本目录收录与分身推荐系统相关的学术论文，按主题分类。

---

## 📂 分类索引

- [01-conversational-memory.md](01-conversational-memory.md) - 对话记忆与检索
- [02-llm-personalization.md](02-llm-personalization.md) - 大模型个性化
- [03-query-understanding.md](03-query-understanding.md) - 查询理解
- [04-recommendation-systems.md](04-recommendation-systems.md) - 推荐系统

---

## 🎯 优先阅读推荐

### P0 级（直接验证核心架构）

| 论文 | 主题 | 相关性 | 文件 |
|-----|------|--------|------|
| **SeCom** (NeurIPS 2024) | 记忆粒度与分段 | ⭐⭐⭐⭐⭐ | [01](01-conversational-memory.md#secom) |
| **PRIME** (2025) | 双记忆模型 | ⭐⭐⭐⭐⭐ | [02](02-llm-personalization.md#prime) |
| **Query Understanding in LLM-based CIS** (2025) | 对话式查询理解 | ⭐⭐⭐⭐⭐ | [03](03-query-understanding.md#llm-cis) |

### P1 级（扩展与优化）

| 论文 | 主题 | 相关性 | 文件 |
|-----|------|--------|------|
| **From Personal to Collective** (2025) | 两层记忆协同 | ⭐⭐⭐⭐ | [02](02-llm-personalization.md#local-global) |
| **Reasoning-enhanced Query Understanding** (2025) | 推理增强 | ⭐⭐⭐⭐ | [03](03-query-understanding.md#reasoning) |
| **Crafting Personalized Agents** (EMNLP 2024) | RAG 个性化 | ⭐⭐⭐⭐ | [02](02-llm-personalization.md#rag-agents) |

### P2 级（参考与启发）

| 论文 | 主题 | 相关性 | 文件 |
|-----|------|--------|------|
| **Memory-Augmented LLM** (2023) | 记忆协调 | ⭐⭐⭐ | [02](02-llm-personalization.md#memory-augmented) |
| **Graph Enhanced BERT** (SIGIR 2023) | 图结构检索 | ⭐⭐⭐ | [03](03-query-understanding.md#graph-bert) |
| **Two-Tower Recommendation** (2025) | 双塔召回 | ⭐⭐⭐ | [04](04-recommendation-systems.md#two-tower) |

---

## 🔍 核心洞察总结

### 记忆粒度
- **Turn-level**: 太细，噪声多
- **Session-level**: 太粗，混合主题
- **Segment-level**: 最优，按主题分段 ✅ (SeCom 验证)

### 双记忆模型
- **Episodic Memory**: 具体交互历史
- **Semantic Memory**: 抽象偏好模式
- 源自认知心理学 ✅ (PRIME 模型)

### 两层协同
- **Local Memory**: 个人历史（千级）
- **Global Memory**: 外部知识（亿级）
- 对应两层检索架构 ✅ (LoGo 框架)

### Query Understanding 演进
- **传统**: 关键词 + 实体
- **大模型时代**: 推理 + 上下文 + 意图分解 ✅

---

_最后更新: 2026-02-21_
