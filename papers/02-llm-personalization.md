# 大模型个性化

LLM Personalization

---

## <a name="prime"></a>PRIME: Large Language Model Personalization with Cognitive Dual-Memory

**⭐ 优先级: P0 | 相关性: ⭐⭐⭐⭐⭐**

### 基本信息
- **发表**: 2025-07-07
- **链接**: https://arxiv.org/abs/2507.04607
- **Semantic Scholar**: https://www.semanticscholar.org/paper/f57b6a7fb7d974f4cd591a4ef6118a7ef4eb54a7

### 核心问题
如何让大模型输出与个人的独特偏好和观点对齐？

### 关键创新：双记忆模型

#### 理论基础：认知心理学
借鉴人类认知系统的双记忆理论：
- **Episodic Memory（情节记忆）**: 具体的事件和经历
- **Semantic Memory（语义记忆）**: 抽象的概念和知识

#### 在 LLM 中的映射

| 认知记忆 | LLM 实现 | 存储内容 | 特征 |
|---------|----------|---------|------|
| **Episodic** | 情节记忆 | 历史交互记录 | 具体、时序性、情境相关 |
| **Semantic** | 语义记忆 | 用户偏好抽象 | 抽象、规律性、可泛化 |

### 架构设计

#### 情节记忆（Episodic Memory）
- **存储**: 历史对话、交互记录、用户行为序列
- **作用**: 提供具体的上下文和案例
- **召回**: 基于相似场景的检索

#### 语义记忆（Semantic Memory）
- **存储**: 用户偏好、决策模式、价值观
- **作用**: 抽象出个性化的"规则"
- **更新**: 从情节记忆中提炼和归纳

### 对本项目的启发

✅ **理论支撑**
- 为「短期记忆 vs 长期记忆」提供认知科学依据
- 情节记忆 → 日常日志（memory/YYYY-MM-DD.md）
- 语义记忆 → 核心洞察（memory/insights.md）

✅ **架构设计**
- 两种记忆的协同工作机制
- 从具体到抽象的记忆演进路径

### 实现思路
1. **情节记忆**: 原始对话记录，按时间存储
2. **语义记忆**: 定期总结、提炼规律、更新偏好
3. **协同召回**: 根据查询类型选择合适的记忆层

---

## <a name="local-global"></a>From Personal to Collective: On the Role of Local and Global Memory

**⭐ 优先级: P1 | 相关性: ⭐⭐⭐⭐**

### 基本信息
- **发表**: 2025-09-28
- **链接**: https://arxiv.org/abs/2509.23767
- **PDF**: https://arxiv.org/pdf/2509.23767

### 核心问题
如何整合个人特定的历史（Local Memory）和外部的集体知识（Global Memory）？

### LoGo 框架（Local-Global Memory）

#### 架构设计
```
User Query
    ↓
┌─────────────┬─────────────┐
│ Local Memory│Global Memory│
│  (个人历史)  │  (外部知识)  │
└─────────────┴─────────────┘
    ↓
  协同融合
    ↓
  LLM 生成
```

#### Local Memory
- **数据源**: 用户历史交互
- **规模**: 千级
- **特点**: 高度个性化、领域特定
- **作用**: 提供个人上下文

#### Global Memory
- **数据源**: 外部知识库、通用知识
- **规模**: 亿级
- **特点**: 广泛、通用、权威
- **作用**: 补充全面性和深度

### 协同机制
1. **Local 优先**: 先检索个人记忆
2. **Global 补充**: 当 Local 不足时，调用 Global
3. **融合生成**: 综合两者生成响应

### 对本项目的启发

✅ **直接对应「两层检索模型」**
- 第一层（快速检索）→ Local Memory（个人记忆）
- 第二层（海量检索）→ Global Memory（外部知识库）

✅ **验证了分层设计的合理性**
- 不同层处理不同规模的数据
- 不同层有不同的召回策略

✅ **实现路径**
- Local Memory: 规则召回 + 轻量排序
- Global Memory: 双塔召回 + LLM 精排

---

## <a name="rag-agents"></a>Crafting Personalized Agents through Retrieval-Augmented

**⭐ 优先级: P1 | 相关性: ⭐⭐⭐⭐**

### 基本信息
- **发表**: EMNLP 2024
- **链接**: https://aclanthology.org/2024.emnlp-main.281.pdf

### 核心贡献
提出通过 RAG（Retrieval-Augmented Generation）来定制个性化 Agent。

### 方法论
1. **构建个人记忆库**: 存储用户的历史交互
2. **检索相关记忆**: 根据当前查询召回相关上下文
3. **增强生成**: 将召回的记忆注入 Prompt

### 实验结果
- 显著提升个性化体验
- 在长期对话中保持连贯性

### 对本项目的启发
- ✅ RAG 是实现个性化的核心技术路径
- ✅ 检索质量直接影响最终效果

---

## <a name="memory-augmented"></a>Memory-Augmented LLM Personalization with Short- and Long-Term Memory

**⭐ 优先级: P2 | 相关性: ⭐⭐⭐**

### 基本信息
- **发表**: 2023-09-20
- **链接**: https://huggingface.co/papers/2309.11696

### 核心观点
区分短期记忆和长期记忆，研究两者的协调机制。

### 短期 vs 长期记忆

| 维度 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| **时间跨度** | 当前会话 | 跨多个会话 |
| **容量** | 有限 | 大容量 |
| **衰减** | 快速 | 缓慢 |
| **作用** | 当前上下文 | 长期偏好 |

### 协调机制
- 短期记忆提供即时上下文
- 长期记忆提供稳定偏好
- 两者动态融合

### 对本项目的启发
- 💡 当前对话 vs 历史记忆的权重平衡
- 💡 记忆的时间衰减策略

---

## 其他相关论文

### Enabling On-Device LLM Personalization
- **链接**: https://dl.acm.org/doi/10.1145/3649329.3655665
- **主题**: 设备端个性化（受限资源下的方案）
- **相关性**: ⭐⭐ 本项目是云端系统，暂不涉及

### Personalized LLM Response Generation with Parameterized Memory
- **链接**: https://www.semanticscholar.org/paper/c1f3c757da46a029ea7fad35c1b183ca460f4100
- **主题**: 参数化记忆注入 + PEFT
- **相关性**: ⭐⭐ 偏向模型训练，本项目主要是检索增强

---

_最后更新: 2026-02-21_
