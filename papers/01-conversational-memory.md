# 对话记忆与检索

Conversational Memory & Retrieval

---

## <a name="secom"></a>SeCom: On Memory Construction and Retrieval for Personalized Conversational Agents

**⭐ 优先级: P0 | 相关性: ⭐⭐⭐⭐⭐**

### 基本信息
- **发表**: NeurIPS 2024 Workshop on Secure and Safe Autonomous Driving
- **作者**: Pan, Wu et al.
- **链接**: https://arxiv.org/abs/2502.05589
- **OpenReview**: https://openreview.net/pdf?id=z8dzEojgvN
- **NeurIPS Virtual**: https://neurips.cc/virtual/2024/104886

### 核心问题
在长期对话中提供连贯且个性化的体验时，现有方法如何通过记忆库检索来增强响应生成？

### 主要贡献

1. **系统研究记忆粒度的影响**
   - Turn-level（对话轮次级）
   - Session-level（会话级）
   - Segment-level（分段级）

2. **揭示现有方法的局限性**
   - Turn-level: 粒度太细，包含大量噪声和冗余信息
   - Session-level: 粒度太粗，混合了多个不相关主题

3. **提出 SeCom 方法**
   - **核心思想**: 在 Segment 级别构建记忆库
   - **技术手段**: 引入对话分段模型（Conversation Segmentation Model）
   - **目标**: 将长期对话按主题划分为连贯的分段

### 方法论

#### 对话分段模型
- 自动识别对话中的主题边界
- 将长对话切分为语义连贯的段落
- 每个段落代表一个独立的主题或讨论

#### 记忆检索流程
1. **构建**: 将对话历史按 Segment 存储
2. **索引**: 为每个 Segment 建立主题标签
3. **召回**: 根据当前查询检索相关 Segment
4. **生成**: 基于召回的 Segment 生成个性化响应

### 实验结果
- Segment-level 记忆显著优于 Turn-level 和 Session-level
- 在长期对话场景下，响应质量和个性化程度均有提升

### 对本项目的启发

✅ **直接验证了我们的架构假设**
- 记忆粒度应该是**主题驱动的 Segment**，而非机械的时间窗口
- 对话分段是记忆构建的关键前置步骤

✅ **实现路径参考**
- 需要一个对话分段模型（或规则）
- Segment 应该包含完整的主题讨论
- 检索时应该以 Segment 为单位，而非单条消息

### 待探索问题
- 如何在个人记忆系统中定义"Segment"？
- 是否可以用简单规则代替模型分段？（千级数据规模）
- 如何处理跨 Segment 的主题关联？

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

## A Memory Fabric for Conversational AI Agents

**⭐ 优先级: P2 | 相关性: ⭐⭐⭐**

### 基本信息
- **发表**: Springer, 2026-02-17
- **链接**: https://link.springer.com/article/10.1007/s44163-026-00992-z
- **主题**: 多 Agent 共享记忆架构

### 核心观点
提出「记忆织物」(Memory Fabric) 架构，支持多个 AI Agent 共享和同步记忆。

### 应用场景
- 多用户协作系统
- Agent 之间的知识迁移
- 团队级别的记忆管理

### 对本项目的启发
- 💡 如果未来扩展到多用户或团队场景，可以参考
- 💡 当前单用户系统可以暂时忽略

---

_最后更新: 2026-02-21_
