# 相关论文

本文档收录与分身推荐系统相关的学术论文。

---

## 核心主题分类

### 1. Conversational Memory & Retrieval

#### On Memory Construction and Retrieval for Personalized Conversational Agents
- **发表**: NeurIPS 2024 (Workshop on SeCom)
- **链接**: https://arxiv.org/abs/2502.05589
- **核心观点**: 
  - 系统研究了记忆粒度对检索增强生成的影响
  - 提出 SeCom 方法：在 segment 级别构建记忆库
  - 引入对话分段模型，将长期对话按主题分割
  - 解决了 turn-level 和 session-level 记忆的局限性
- **相关性**: ⭐⭐⭐⭐⭐ 直接对应我们的「记忆粒度」和「检索召回」问题

#### Memory in the Age of AI Agents: A Survey
- **GitHub**: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- **核心观点**: AI Agent 记忆系统的综合调研
- **相关性**: ⭐⭐⭐⭐ 提供了全景视角和文献索引

#### A Memory Fabric for Conversational AI Agents
- **发表**: Springer (2026)
- **链接**: https://link.springer.com/article/10.1007/s44163-026-00992-z
- **核心观点**: 
  - 提出「记忆织物」架构，支持多 Agent 共享记忆
  - 解决大模型对话系统的记忆管理问题
- **相关性**: ⭐⭐⭐ 提供了多用户/多 Agent 场景的启发

---

### 2. LLM Personalization

#### PRIME: Large Language Model Personalization with Cognitive Dual-Memory
- **发表**: 2025-07-07
- **链接**: https://arxiv.org/abs/2507.04607
- **核心观点**: 
  - 将认知心理学的「双记忆模型」引入 LLM 个性化
  - Episodic Memory（情节记忆）→ 历史交互
  - Semantic Memory（语义记忆）→ 用户偏好抽象
- **相关性**: ⭐⭐⭐⭐⭐ 直接对应我们的「短期/长期记忆」设计

#### From Personal to Collective: On the Role of Local and Global Memory
- **发表**: 2025-09-28
- **链接**: https://arxiv.org/abs/2509.23767
- **核心观点**: 
  - 研究 Local Memory（用户特定历史）和 Global Memory（集体知识）的协同
  - 提出 LoGo 框架，整合个人和全局记忆
- **相关性**: ⭐⭐⭐⭐ 对应我们的「两层检索模型」（个人记忆 + 外部知识库）

#### Memory-Augmented LLM Personalization with Short- and Long-Term Memory
- **发表**: 2023-09-20
- **链接**: https://huggingface.co/papers/2309.11696
- **核心观点**: 
  - 区分短期记忆和长期记忆
  - 研究两者的协调机制
- **相关性**: ⭐⭐⭐⭐ 提供记忆分层的理论支持

#### Crafting Personalized Agents through Retrieval-Augmented
- **发表**: EMNLP 2024
- **链接**: https://aclanthology.org/2024.emnlp-main.281.pdf
- **核心观点**: 
  - 提出通过检索用户个人记忆来定制 Agent
  - 利用 RAG 增强个性化体验
- **相关性**: ⭐⭐⭐⭐ 与我们的检索增强方案一致

---

### 3. Query Understanding & Information Retrieval

#### Reasoning-enhanced Query Understanding through Decomposition and Interpretation
- **发表**: 2025-09-08
- **链接**: https://arxiv.org/abs/2509.06544
- **核心观点**: 
  - 通过分解和解释来增强查询理解
  - 利用推理能力理解复杂查询意图
- **相关性**: ⭐⭐⭐⭐ 对应我们的「Query Understanding」层

#### Query Understanding in the Age of Large Language Models
- **发表**: 2023-06-28
- **链接**: https://arxiv.org/abs/2306.16004
- **核心观点**: 
  - 大模型时代的查询理解新范式
  - 自然语言控制搜索和信息检索界面
- **相关性**: ⭐⭐⭐⭐ 提供大模型时代的 Query Understanding 综述

#### Query Understanding in LLM-based Conversational Information Seeking
- **发表**: 2025-04-09
- **链接**: https://arxiv.org/abs/2504.06356
- **核心观点**: 
  - 对话式信息检索中的查询理解
  - 上下文感知的意图解释
  - 解决歧义和动态查询
- **相关性**: ⭐⭐⭐⭐⭐ 直接对应我们的「对话上下文 + 意图理解」需求

#### Graph Enhanced BERT for Query Understanding
- **发表**: SIGIR 2023
- **链接**: https://arxiv.org/abs/2204.06522
- **核心观点**: 
  - 利用图结构增强查询理解
  - 探索用户搜索意图
- **相关性**: ⭐⭐⭐ 提供图结构召回的启发

#### DeepRetrieval: Powerful Query Generation for Information Retrieval with Reinforcement Learning
- **发表**: 2025-02-28
- **链接**: https://www.researchgate.net/publication/389547553
- **核心观点**: 
  - 利用大模型通过查询增强提升检索性能
  - 强化学习优化查询生成
- **相关性**: ⭐⭐⭐ 提供查询优化的技术路径

---

### 4. Recommendation Systems (待补充)

_(需要进一步搜索推荐系统相关论文，特别是双塔召回、多路召回、Query Understanding 在推荐中的应用)_

---

## 核心洞察总结

### 记忆粒度
- **Turn-level**: 太细粒度，噪声多
- **Session-level**: 太粗粒度，混合多个主题
- **Segment-level**: 最优粒度，按主题分段（SeCom 论文验证）

### 双记忆模型
- **Episodic Memory（情节记忆）**: 具体的交互历史
- **Semantic Memory（语义记忆）**: 抽象的偏好和模式
- 类比认知心理学的记忆机制（PRIME 论文）

### Local vs Global Memory
- **Local Memory**: 个人特定的历史和偏好
- **Global Memory**: 外部知识库和集体智慧
- 两者需要协同工作（LoGo 框架）

### Query Understanding 演进
- **传统**: 关键词匹配、实体识别
- **大模型时代**: 推理增强、上下文感知、意图分解
- **对话式**: 动态查询、歧义消解、连续性理解

---

## 待深入阅读

### 优先级 P0
1. **SeCom** (NeurIPS 2024) - 记忆粒度和分段方法
2. **PRIME** (2025) - 双记忆模型的具体实现
3. **Query Understanding in LLM-based CIS** (2025) - 对话式查询理解

### 优先级 P1
4. **From Personal to Collective** (2025) - Local/Global Memory 协同
5. **Reasoning-enhanced Query Understanding** (2025) - 推理增强的查询理解
6. **Crafting Personalized Agents** (EMNLP 2024) - RAG 个性化方案

### 优先级 P2
7. **Memory-Augmented LLM Personalization** (2023) - 短期/长期记忆协调
8. **Graph Enhanced BERT** (SIGIR 2023) - 图结构增强
9. **Memory in the Age of AI Agents** - 综合调研

---

## 搜索关键词记录

- `conversational memory AI agent`
- `LLM memory personalization`
- `query understanding information retrieval`
- `personalized recommendation dual tower recall` (待搜索)
- `multi-stage retrieval ranking` (待搜索)

---

_最后更新: 2026-02-21_
