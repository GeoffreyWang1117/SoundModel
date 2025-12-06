# 后续实验计划

## 当前实验完成情况总结

### ✅ 已完成实验

| 实验 | 目标 | 结果 | 关键发现 |
|------|------|------|----------|
| Exp 1 | 概念验证 | 成功 | 管道运行正常 |
| Exp 2-5 | 数据规模验证 | 成功 | 1000样本最佳 |
| Exp 6 | 文学 vs 合成数据 | 成功 | 文学数据性能更好 |
| Exp 7 | 特征对比 | **突破** | Acoustic (46D) 40.45% 优于 WavLM 27.82% |
| Exp 8 | 消融研究 | 成功 | 频谱25.71% + 韵律14.74% 协同效应 |
| Exp 9 | 跨说话人测试 | **失败** | 38.5× 性能下降 (3.0502 vs 0.0792) |
| Exp 10 | 相对特征归一化 | **失败** | 更差 (3.3155) |
| Exp 11 | 2说话人训练 | 部分成功 | 16.4% 改善，但仍差32.2× |

### 🔑 核心问题

**跨说话人泛化严重失败** - 这是当前最关键的研究障碍：
- 单说话人训练：38.5× 性能下降
- 2说话人训练：仍然32.2× 性能下降
- 同说话人验证具有误导性：10.3× 差距

---

## 优先级1：快速验证实验（1-3天）

### Exp 12a: Acoustic特征跨说话人测试 🔥

**动机**: Exp 7显示Acoustic特征在域内性能超过WavLM，但未测试跨说话人

**假设**: Acoustic特征可能在跨说话人场景下也更优

**实验设计**:
```
训练: Claribel训练集 (1000样本) - Acoustic 46D
测试: Damien测试集 (100样本) - Acoustic 46D
比较: vs Exp 9a (WavLM跨说话人: 3.0502)
```

**预期结果**:
- 如果Acoustic < 3.0: 说明Acoustic特征跨说话人更好
- 如果Acoustic > 3.0: 说明问题不在特征类型

**时间**: 
- 提取Damien的Acoustic特征: 10分钟
- 评估: 5分钟
- **总计: 15分钟**

**优先级**: ⭐⭐⭐⭐⭐ (最高，快速且关键)

---

### Exp 12b: Fusion特征跨说话人测试

**动机**: Exp 7显示Fusion (302D) 域内性能0.0771，介于Acoustic和WavLM之间

**实验设计**:
```
训练: 使用Exp 7的Fusion模型
测试: Damien测试集
比较: vs Acoustic和WavLM跨说话人性能
```

**预期**: Fusion可能结合两者优势

**时间**: 10分钟

**优先级**: ⭐⭐⭐⭐

---

### Exp 12c: 2说话人Acoustic特征训练

**动机**: Exp 11使用WavLM实现16.4%改善，Acoustic是否更好？

**实验设计**:
```
训练: Claribel (500) + Damien (50) - Acoustic 46D
测试: Andrew - Acoustic 46D
比较: vs Exp 11 (WavLM 2说话人: 2.5503)
```

**预期**: 如果Acoustic更speaker-invariant，应该 < 2.5503

**时间**: 
- 提取特征: 30分钟
- 训练: 30分钟
- **总计: 1小时**

**优先级**: ⭐⭐⭐⭐

---

## 优先级2：扩展多说话人实验（3-7天）

### Exp 13: 4说话人平衡训练

**动机**: Exp 11的2说话人不足，且存在10:1不平衡

**实验设计**:
```
训练数据:
  - Claribel Dervla (女): 200样本
  - Damien Black (男): 200样本
  - Andrew Chipper (男): 200样本  
  - Gracie Wise (女): 200样本
  - 总计: 800样本，性别平衡

验证集:
  - 从每个说话人抽取20样本: 80样本

测试集:
  - Viktor Eka (男): 100样本 (完全未见)

特征: WavLM 256D (后续可测试Acoustic)
```

**预期结果**:
- 目标: < 2.0 cross-speaker loss
- 性别平衡应该改善泛化

**时间**: 
- 数据生成: 4小时 (800样本)
- 训练: 40分钟
- **总计: 5小时**

**优先级**: ⭐⭐⭐⭐

---

### Exp 14: 6说话人大规模训练

**动机**: 进一步增加说话人多样性

**实验设计**:
```
训练数据:
  - 6 speakers × 150 samples = 900 total
  - 女性: Claribel, Gracie, Sofia Hellen
  - 男性: Damien, Andrew, Viktor

测试集:
  - 2个完全未见说话人 × 100样本
```

**预期**: 接近 < 1.5

**时间**: 6小时

**优先级**: ⭐⭐⭐

---

## 优先级3：架构改进实验（1-2周）

### Exp 15: 说话人自适应归一化

**动机**: 显式建模说话人特性

**方法**:
```python
class SpeakerAdaptiveProjection(nn.Module):
    def __init__(self, emotion_dim, speaker_emb_dim, hidden_dim):
        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim)
        
        # Speaker-specific scale and shift
        self.scale_net = nn.Linear(speaker_emb_dim, hidden_dim)
        self.shift_net = nn.Linear(speaker_emb_dim, hidden_dim)
    
    def forward(self, emotion_feat, speaker_emb):
        emotion = self.emotion_proj(emotion_feat)
        
        # Adaptive normalization
        scale = self.scale_net(speaker_emb)
        shift = self.shift_net(speaker_emb)
        
        normalized = emotion * scale + shift
        return normalized
```

**说话人嵌入提取**:
- 使用X-vectors或ECAPA-TDNN
- 预训练的说话人识别模型

**时间**: 2-3天

**优先级**: ⭐⭐⭐

---

### Exp 16: 对抗性说话人解耦

**动机**: 强制情感特征与说话人无关

**方法**:
```python
# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# 训练
emotion_feat = emotion_projection(audio)
llm_output = llm(emotion_feat)
llm_loss = CrossEntropy(llm_output, target)

# Adversarial speaker classification
reversed_feat = GRL.apply(emotion_feat, lambda_grl)
speaker_pred = speaker_classifier(reversed_feat)
speaker_loss = CrossEntropy(speaker_pred, speaker_id)

# Total loss
loss = llm_loss - lambda_adv * speaker_loss
```

**预期**: 显著降低说话人依赖性

**时间**: 3-5天

**优先级**: ⭐⭐⭐

---

## 优先级4：数据改进（长期）

### Exp 17: 真实人类语音

**数据源**:
- IEMOCAP: 情感对话数据集
- MSP-Podcast: 自然情感语音
- MELD: 多模态情感数据

**挑战**:
- 需要对齐文本-音频-情感标签
- 数据清洗和预处理

**时间**: 1-2周

**优先级**: ⭐⭐

---

## 推荐执行顺序

### 第一阶段：快速验证（今天-明天）

1. **Exp 12a: Acoustic跨说话人** (15分钟) ⭐⭐⭐⭐⭐
   - 立即执行，验证Acoustic是否更speaker-invariant
   
2. **Exp 12b: Fusion跨说话人** (10分钟) ⭐⭐⭐⭐
   - 顺便测试，了解Fusion特性
   
3. **根据12a/12b结果决定**:
   - 如果Acoustic更好 → 执行Exp 12c (1小时)
   - 如果差不多 → 直接进入第二阶段

### 第二阶段：多说话人扩展（2-3天）

4. **Exp 13: 4说话人平衡训练** (5小时) ⭐⭐⭐⭐
   - 如果Exp 12c的Acoustic好，用Acoustic
   - 否则用WavLM
   
5. **分析Exp 13结果**:
   - 如果 < 2.0: 说明规模有效 → 继续Exp 14
   - 如果 > 2.5: 说明需要架构改进 → 跳到第三阶段

### 第三阶段：架构改进（1-2周）

6. **Exp 15: 说话人自适应归一化** (2-3天) ⭐⭐⭐
7. **Exp 16: 对抗性解耦** (3-5天) ⭐⭐⭐

### 第四阶段：数据改进（可选）

8. **Exp 17: 真实人类语音** (1-2周) ⭐⭐

---

## 关键问题跟踪

### 待回答的科学问题

1. **Acoustic vs WavLM 跨说话人性能**
   - Exp 12a将回答此问题
   - 如果Acoustic更好，说明interpretable features更speaker-invariant
   
2. **说话人数量 vs 性能**
   - Exp 11 (2说话人): 2.5503
   - Exp 13 (4说话人): 预期 < 2.0
   - Exp 14 (6说话人): 预期 < 1.5
   - 绘制scaling curve
   
3. **架构 vs 数据**
   - 是否需要显式disentanglement？
   - 还是仅需更多说话人数据？

### 成功标准

- **最低标准**: Cross-speaker loss < 1.0
- **理想标准**: Cross-speaker loss < 0.5 (接近in-domain 0.0792的6倍)
- **论文补充**: 新实验结果可以作为supplementary material

---

## 立即行动

**建议现在执行**: Exp 12a (15分钟)

这是最快且最关键的验证实验，将指导后续方向。

是否开始执行Exp 12a?
