# 实验 10b 结果：相对声学特征失败分析

**日期**: 2025-11-15
**实验目标**: 使用相对声学特征（Relative Acoustic Features）解决跨说话人泛化问题
**结果**: ❌ **失败** - 跨说话人性能未改善，反而略有下降

---

## 一、实验设置

### 1.1 特征设计（44D）

**Prosodic特征 (13D)**:
- 相对范围: `pitch_range_rel = (p95 - p5) / median`
- 变异系数: `pitch_cv = std / mean`
- 归一化斜率: `pitch_slope_rel = slope / mean`
- 百分位比率: `pitch_p90_p10_ratio = p90 / p10`
- 偏度: `pitch_skewness`
- 能量特征: `energy_range_rel`, `energy_cv`, `energy_peak_mean_ratio`, `energy_dynamic_range_db`
- 时间特征: `duration`, `speech_rate_rel`, `zcr_cv`

**Spectral特征 (31D)**:
- 13个MFCC均值
- 13个Delta MFCC均值
- 5个光谱形状特征: `spectral_centroid_mean/cv`, `spectral_rolloff_mean`, `spectral_bandwidth_mean`, `spectral_flux_mean/cv`

**数学原理**:
- 使用**无量纲比率**代替绝对值
- 变异系数 CV = σ/μ：尺度无关的变异度量
- 相对范围：归一化至中位数

### 1.2 训练配置

- **模型**: Qwen2.5-1.5B + LoRA (r=16, α=32) + 4-bit量化
- **数据**: train_1000 (800训练/200验证)
- **训练**: 5 epochs, batch_size=4, lr=5e-5
- **特征**: 44D relative acoustic embeddings

---

## 二、实验结果

### 2.1 训练过程

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 1.0419    | 0.1566   |
| 2     | 0.1423    | 0.1327   |
| 3     | 0.1224    | 0.1223   |
| 4     | 0.1259    | 0.1053   |
| 5     | 0.1058    | **0.0949** |

**最佳验证loss**: 0.0949
**训练状态**: ✅ 收敛良好，无过拟合

### 2.2 跨说话人测试

**测试集**: test_cross_speaker_damien (100 samples)
**测试loss**: **3.3155**

### 2.3 与基线对比

| 实验 | 特征类型 | 域内Val Loss | 跨说话人Test Loss | 泛化倍数 |
|------|---------|-------------|------------------|---------|
| Exp 7b | 绝对声学 (46D) | 0.0792 | - | - |
| Exp 9a | 绝对声学 (46D) | 0.0912 | 3.0502 | **33.5×** |
| **Exp 10b** | **相对声学 (44D)** | **0.0949** | **3.3155** | **34.9×** |

**结论**: ❌ 相对特征未能改善跨说话人泛化，反而略有下降（+0.27）

---

## 三、失败原因分析

### 3.1 为什么相对特征失败？

#### 问题1: 相对特征仍保留speaker-specific信息

虽然使用了CV、比率等"无量纲"特征，但speaker-dependent patterns仍然存在：

1. **CV的speaker依赖性**:
   ```
   CV = σ/μ

   对于pitch：
   - 高水平voice actor: CV可能更大（表达更丰富）
   - 低水平voice actor: CV可能更小（表达单一）
   - TTS: CV非常稳定（机械生成）
   ```

2. **相对范围的speaker依赖性**:
   ```
   pitch_range_rel = (p95 - p5) / median

   不同speaker的median本身不同：
   - 女性: median ~ 200-250 Hz
   - 男性: median ~ 100-120 Hz

   即使归一化，不同性别的range_rel分布仍然不同
   ```

3. **MFCC的speaker依赖性**:
   - MFCC编码了声道特征（formants）
   - 声道长度是speaker-specific的物理属性
   - 即使取均值，不同speaker的MFCC分布仍然截然不同

#### 问题2: 情感表达的speaker差异

不同speaker在表达同一情感时，使用的prosodic strategies不同：

```
示例：表达"愤怒"

高水平voice actor:
- pitch_cv = 0.35 (大幅变化)
- energy_peak_mean_ratio = 2.5 (强调峰值)
- speech_rate_rel = 1.3 (语速加快)

普通speaker:
- pitch_cv = 0.15 (变化较小)
- energy_peak_mean_ratio = 1.5 (峰值不明显)
- speech_rate_rel = 1.1 (语速略快)

TTS:
- pitch_cv = 0.05 (几乎不变)
- energy_peak_mean_ratio = 1.2 (峰值很小)
- speech_rate_rel = 1.0 (无变化)
```

#### 问题3: 训练数据的speaker homogeneity

- 训练集1000个样本全部来自**单一speaker**（Claribel Dervla）
- 模型学习的是"Claribel如何用prosody表达情感"
- 测试集使用不同speaker（Damien Black）时，即使用相对特征，表达模式仍然不匹配

### 3.2 数学分析：为什么CV不足以消除speaker identity？

假设两个speaker的F0分布：

**Speaker A (女性)**:
```
μ_A = 220 Hz, σ_A = 40 Hz
CV_A = 40/220 = 0.182

某个情感样本:
μ_emotion = 250 Hz, σ_emotion = 50 Hz
CV_emotion = 50/250 = 0.200
```

**Speaker B (男性)**:
```
μ_B = 110 Hz, σ_B = 20 Hz
CV_B = 20/110 = 0.182

相同情感样本:
μ_emotion = 125 Hz, σ_emotion = 25 Hz
CV_emotion = 25/125 = 0.200
```

看起来CV相同（0.200），但实际上：

1. **统计假设失效**: CV假设数据服从正态分布，但pitch contour通常是非高斯的
2. **时序信息丢失**: CV是静态聚合特征，丢失了pitch contour的temporal dynamics
3. **上下文依赖**: 相同的CV值在不同speaker的baseline上可能对应不同的情感

**关键洞察**: Speaker normalization需要的不是**feature-level normalization**，而是**distribution-level alignment**

---

## 四、为什么Exp 10b比Exp 9a更差？

### 4.1 Bug修复的影响

**Exp 9a的实际配置** (发现evaluation script bug后):
- 实际上是**text-only**模式
- emotion_embeddings = None (因为key名称错误: 'emotion_embedding' vs 'emotion_embeddings')
- Test loss: 3.0502

**Exp 10b的配置**:
- 正确使用了**相对特征** (44D)
- Test loss: 3.3155

**结论**: 相对特征不仅没帮助，还略微损害了性能

### 4.2 可能的解释

1. **信息损失**: 44D相对特征可能丢失了一些对emotion有用的信息
2. **特征质量**: 部分相对特征（如formant相关）提取失败率高
3. **训练不匹配**: 模型在相对特征上训练，但这些特征与emotion的关联在跨speaker时仍然break

---

## 五、关键发现

### 5.1 相对特征方法的局限性

✅ **成功**:
- 44D相对特征提取成功率100%
- 训练收敛良好（Val loss 0.0949）
- 无过拟合现象

❌ **失败**:
- 跨说话人泛化**完全失败**（34.9×退化）
- 甚至比text-only更差（+0.27）

### 5.2 Speaker Normalization的真正挑战

**相对特征方法失败的根本原因**:

1. **Prosodic strategies的speaker差异** > 特征归一化能消除的差异
2. **情感表达的个体化** > 统计归一化的能力
3. **需要更激进的方法**: 不是feature normalization，而是**distribution alignment**或**multi-speaker training**

---

## 六、下一步实验方向

### 6.1 放弃的方向

❌ **Exp 10a: Z-score Normalization**
- 基于相对特征失败的分析，z-score也不太可能成功
- z-score仍然是feature-level normalization，无法解决distribution mismatch

### 6.2 推荐的方向

#### 方向1: Multi-Speaker Training (Exp 11)

**思路**: 让模型在训练时就见到多个speaker

**实施**:
```python
# 使用全部EmoV-DB数据（4 speakers）
speakers = ['bea', 'jenie', 'josh', 'sam']
train_data = concatenate([
    load_speaker_data(spk, n_samples=250)
    for spk in speakers
])  # Total: 1000 samples from 4 speakers

# 训练时speaker ID作为conditioning
model_input = {
    'text': text,
    'emotion': emotion_embedding,
    'speaker_id': speaker_onehot  # 4D one-hot
}
```

**期待效果**: 模型学习speaker-invariant的emotion patterns

#### 方向2: Domain Adaptation (Exp 12)

**思路**: 使用unsupervised adaptation将Damien的特征对齐到Claribel

**实施**:
```python
# 1. Extract speaker statistics
stats_claribel = compute_statistics(train_1000)
stats_damien = compute_statistics(test_cross_speaker)

# 2. Apply distribution matching
damien_adapted = match_distribution(
    damien_features,
    source_stats=stats_damien,
    target_stats=stats_claribel,
    method='histogram_matching'  # or 'CycleGAN', 'CORAL'
)

# 3. Test with adapted features
```

#### 方向3: Temporal Dynamics (Exp 13)

**思路**: 不用静态聚合，而是保留完整的prosodic contours

**实施**:
```python
# 提取pitch/energy contours (时序)
pitch_contour = extract_pitch_contour(audio)  # [T,]
energy_contour = extract_energy_contour(audio)  # [T,]

# 使用DCT编码contour shape (speaker-invariant)
pitch_dct = dct(pitch_contour, n_coeffs=10)  # [10,]
energy_dct = dct(energy_contour, n_coeffs=10)  # [10,]

# 相对于speaker baseline
pitch_dct_relative = pitch_dct / speaker_pitch_baseline
```

**期待效果**: Contour shape比静态统计量更speaker-invariant

### 6.3 优先级排序

1. **最优先**: Exp 11 (Multi-Speaker Training)
   - 最直接解决根本问题
   - 需要重新生成数据（4 speakers × 250 samples）

2. **次优先**: Exp 13 (Temporal Dynamics)
   - 理论上更speaker-invariant
   - 实现相对简单

3. **备选**: Exp 12 (Domain Adaptation)
   - 技术难度较高
   - 需要选择合适的adaptation方法

---

## 七、总结

### 7.1 实验结论

**Exp 10b**展示了一个**失败但有价值**的尝试：

✅ 证明了相对特征方法的不足
✅ 揭示了speaker normalization的真正挑战
✅ 为后续实验提供了重要insights

### 7.2 关键Insights

1. **Feature-level normalization不足以消除speaker identity**
2. **Speaker differences存在于prosodic strategies层面**，不仅仅是绝对值差异
3. **需要更根本的解决方案**: multi-speaker training或distribution alignment

### 7.3 下一步行动

**建议**: 进行**Exp 11: Multi-Speaker Training**

**理由**:
- 最直接解决speaker泛化问题
- 让模型学习speaker-invariant emotion representations
- 符合实际应用场景（需要支持多个user）

**实施计划**:
1. 生成multi-speaker训练数据（4 speakers × 250 samples）
2. 添加speaker conditioning机制
3. 在held-out speaker上测试泛化能力

---

**实验10b状态**: ✅ 完成（失败但informative）
**下一步**: 🔄 设计并执行Exp 11
