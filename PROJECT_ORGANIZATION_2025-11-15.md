# 项目整理总结 - 2025-11-15

## 📁 当前项目结构

```
Emotional-Aware-Reasoning/
├── audio_augmented_llm/               # 核心代码库
│   ├── src/
│   │   ├── tts_pipeline/              # TTS语音合成
│   │   │   └── xtts_synthesizer.py    # XTTS v2封装
│   │   ├── emotion_encoder/           # 情感特征提取
│   │   │   ├── acoustic_features.py   # ✅ 46D声学特征 (当前最佳)
│   │   │   ├── wavlm_encoder.py       # WavLM基线
│   │   │   └── [TODO] normalized_features.py  # Speaker归一化
│   │   └── student_training/          # 学生模型训练
│   │       ├── student_model.py       # QLoRA + emotion integration
│   │       └── dataset.py             # 数据加载器
│   ├── data/                          # 数据集 (不在git)
│   │   ├── train_1000/                # ✅ 主训练集 (1000样本)
│   │   │   ├── metadata.json
│   │   │   ├── audio/                 # 1000 wav files
│   │   │   ├── acoustic_embeddings.npz  # 46D特征
│   │   │   ├── prosodic_only.npz      # 13D (Exp 8a)
│   │   │   ├── spectral_only.npz      # 30D (Exp 8b)
│   │   │   └── no_prosodic.npz        # 33D (Exp 8c)
│   │   ├── test_cross_speaker_damien/ # ✅ 跨speaker测试集
│   │   │   ├── metadata.json
│   │   │   ├── audio/                 # 100 wav files (Damien Black)
│   │   │   └── acoustic_embeddings.npz
│   │   └── [FUTURE] train_multispeaker_1000/
│   └── models/                        # 训练好的模型
│       ├── exp7b_acoustic_1000/       # ✅ 最佳模型 (Val: 0.0792)
│       │   └── best_model/
│       │       ├── adapter_model.safetensors  # LoRA权重
│       │       ├── emotion_projection.pt      # Emotion层
│       │       └── training_config.json
│       ├── exp8_prosodic_only/        # Ablation: 13D
│       ├── exp8_spectral_only/        # Ablation: 30D
│       └── exp8_no_prosodic/          # Ablation: 33D
│
├── scripts/                           # 实验脚本
│   ├── generate_teacher_data.py       # 生成synthetic对话数据
│   ├── generate_cross_speaker_test.py # ✅ 生成跨speaker测试集
│   ├── extract_acoustic_features_batch.py  # ✅ 批量特征提取
│   ├── extract_feature_subset.py      # ✅ 提取特征子集 (ablation)
│   ├── train_student_simple.py        # ✅ 主训练脚本
│   ├── evaluate_model.py              # ✅ 模型评估
│   ├── run_ablation_exp8.sh           # ✅ Ablation批处理
│   └── [TODO] extract_normalized_features.py  # Speaker归一化
│
├── paper/                             # LaTeX论文
│   ├── main.tex                       # 主文件
│   ├── sections/
│   │   ├── abstract.tex               # ✅ 完成
│   │   ├── introduction.tex           # ✅ 完成
│   │   ├── related_work.tex           # ⏳ TODO: 添加引用
│   │   ├── methodology.tex            # ✅ 完成
│   │   ├── experimental_setup.tex     # ✅ 完成
│   │   ├── results.tex                # ✅ 完成
│   │   ├── analysis.tex               # ✅ 更新 (Exp 8, 9a)
│   │   ├── discussion.tex             # ⏳ TODO
│   │   └── conclusion.tex             # ✅ 完成
│   └── references.bib                 # ⏳ TODO: 补充文献
│
├── outputs/                           # 训练日志
│   ├── exp7b_acoustic_1000_log.txt    # ✅ 最佳模型训练日志
│   ├── exp8a_prosodic_only_log.txt    # ✅ Ablation结果
│   ├── exp8b_spectral_only_log.txt
│   ├── exp8c_no_prosodic_log.txt
│   └── exp9a_cross_speaker_eval_log.txt  # ✅ 泛化测试
│
├── 文档/                              # 项目文档
│   ├── README.md                      # ✅ 主文档 (已更新)
│   ├── EXPERIMENT_LOG.md              # 详细实验记录
│   ├── EXPERIMENT_9_RESULTS.md        # Exp 9a分析
│   ├── EXPERIMENT_9_SUMMARY.md        # Exp 9a总结
│   ├── NEXT_EXPERIMENTS_PLAN.md       # ✅ 下一步计划
│   ├── PROGRESS_SUMMARY.md            # 项目进度
│   ├── PROGRESS_UPDATE_2025-11-15_v2.md  # 最新进展
│   └── PROJECT_ORGANIZATION_2025-11-15.md  # 本文件
│
└── [配置文件]
    ├── .gitignore                     # Git忽略规则
    ├── requirements.txt               # Python依赖
    └── environment.yml                # Conda环境

总计文件统计:
- Python脚本: 15个
- LaTeX文件: 10个
- 数据集: 2个 (1100样本)
- 模型checkpoints: 5个
- 日志文件: 5个
- 文档: 9个
```

## 🔥 核心实验成果总结

### 1. Experiment 7b: Acoustic Features突破

**配置**:
- 特征: 46D acoustic features (prosodic 13D + spectral 30D + formants 3D)
- 模型: Qwen2.5-1.5B + QLoRA (r=16, α=32)
- 数据: 1000样本，5 epochs
- Speaker: Claribel Dervla (female, XTTS v2)

**结果**:
```
Text-Only baseline:  0.1330
WavLM (256D):        0.0960  (+27.82%)
Acoustic (46D):      0.0792  (+40.45%)  ⭐ BEST

效率优势:
- 17.5% better than WavLM
- 8.06× dimensional efficiency (0.88% vs 0.109% per dimension)
```

### 2. Experiment 8: Feature Ablation

**目的**: 识别哪些特征贡献最大

**结果**:
| Feature Set | Dim | Val Loss | Contribution |
|-------------|-----|----------|-------------|
| Prosodic only | 13D | 0.1233 | 7.29% |
| Spectral only | 30D | 0.1228 | 7.67% |
| w/o Prosodic | 33D | 0.0988 | 25.71% |
| **Full** | **46D** | **0.0792** | **40.45%** |

**关键发现**: **Feature Synergy**
- Spectral是骨干: 25.71%贡献
- Prosodic提供协同: +14.74%额外增益
- 协同奖励: +7.45% (40.45% - 33.00%)
- 结论: 特征间存在非线性交互

### 3. Experiment 9a: Cross-Speaker Generalization (⚠️ 关键负面结果)

**测试**:
- Training: Claribel Dervla (female, 220 Hz平均F0)
- Test: Damien Black (male, 120 Hz平均F0)
- 样本: 100 (相同文本，不同speaker)

**结果**:
```
In-domain (Claribel):     0.0792
Cross-speaker (Damien):   3.0502  (+3752%)

性能下降: 38.5× worse
结论: 灾难性泛化失败
```

**根本原因分析**:
1. **绝对特征speaker-dependent**
   - Female F0: 200-400 Hz
   - Male F0: 100-150 Hz
   - Speaker差异 >> Emotion差异
   - SNR_emotion ≈ 0.16 (emotion被speaker淹没)

2. **缺少归一化**
   - 未做per-speaker z-score标准化
   - 模型学到speaker identity而非emotion

3. **静态聚合丢失时序动态**
   - Mean/Std无法区分上升vs下降轮廓
   - 情感在时序变化，而非静态值

## 🔬 声学物理模型应用总结

### 1. Source-Filter Model (Fant 1960)

**数学模型**:
```
S(ω) = E(ω) · H(ω)

E(ω): 声门激励 (决定F0)
H(ω): 声道滤波器 (决定formants)
```

**应用到情感**:
- F0 ↔ Arousal (唤醒度)
- F1, F2, F3 ↔ Tension/Valence (紧张度/价态)
- Energy (RMS) ↔ Intensity (强度)

### 2. MFCC的心理声学基础

**Mel scale**:
```
m = 2595 · log₁₀(1 + f/700)
```

模拟人耳非线性频率感知:
- 低频敏感 (语音基频区域)
- 高频分辨率降低

**情感编码**:
- 低阶MFCC (1-5): 谱包络 → 音色 → 情感基调
- 高阶MFCC (6-13): 谱细节 → 粗糙度 → 唤醒度
- Delta MFCC: 时序变化 → 韵律动态

### 3. YIN Algorithm for F0

**数学原理**:
```
差分函数: d_t(τ) = Σ(x_j - x_{j+τ})²
归一化: d'_t(τ) = d_t(τ) / [(1/τ)Σd_t(j)]
```

优于autocorrelation:
- 避免能量偏差
- 更鲁棒的pitch检测

### 4. LPC for Formants

**全极点模型**:
```
H(z) = G / (1 - Σa_k·z^(-k))
```

**共振峰提取**:
- 求解多项式根找极点
- 极点频率 = 共振峰频率
- F1, F2位置反映声道形状

**情感关联**:
- F1高 → 张开度大 → 兴奋/惊讶
- F1低 → 张开度小 → 悲伤/恐惧
- F2-F1距离 → 发音清晰度

## 🎯 下一步实验路线图

### Phase 4A: Speaker Normalization (优先级：最高)

**目标**: 消除speaker identity bias，提升cross-speaker泛化

#### Experiment 10a: Z-Score Normalization
```python
F0_norm = (F0 - μ_speaker) / σ_speaker
```

**预期**:
- Cross-speaker: 3.0502 → <0.5 (90%改善)
- In-domain: 保持 ~0.08

**时间**: 3-4天
- Day 1: 实现speaker statistics计算
- Day 2: 实现归一化特征提取
- Day 3: 训练和评估
- Day 4: 分析结果

#### Experiment 10b: Relative Features
```python
F0_range_rel = (F0_max - F0_min) / F0_median
```

**优势**: 无需speaker statistics
**预期**: Cross-speaker <1.0
**时间**: 2-3天

### Phase 4B: Temporal Dynamics (优先级：高)

#### Experiment 11a: DCT Pitch Contours
```python
dct_coeffs = dct(f0_normalized, type=2)[:8]
```

**捕捉**:
- c1: 整体趋势 (上升/下降)
- c2: 加速度 (平滑/急促)
- c3-8: 微小变化 (嗓音质量)

**时间**: 3-4天

#### Experiment 11b: Multi-Scale Features
```python
- Phone level (50ms): 微观变化
- Syllable level (200ms): 韵律节奏
- Utterance level (2s): 整体趋势
```

**时间**: 2-3天

### Phase 4C: Multi-Speaker Training (优先级：中)

#### Experiment 12: 4-Speaker Dataset
```
Claribel Dervla (F) + Damien Black (M) +
Gilberto Mathias (M) + Royston Min (M)

4 speakers × 250 samples = 1000 total
```

**强制模型学习**: Speaker-invariant emotion patterns

**时间**: 5-6天

### Phase 4D: Hybrid (优先级：低)

#### Experiment 13: Normalized Acoustic + WavLM
```
Concat[
  Acoustic_normalized (46D),
  WavLM (256D)
] = 302D fusion
```

**预期**: 结合interpretability + learned patterns
**时间**: 2-3天

## 📊 完整结果对比表

| Exp | Method | Dim | Train Samples | Val Loss | Cross-Speaker | Efficiency |
|-----|--------|-----|---------------|----------|---------------|------------|
| Baseline | Text-Only | 0 | 1000 | 0.1330 | - | - |
| 5 | WavLM | 256D | 1000 | 0.0960 | ? | 0.109%/dim |
| 7b | Acoustic | 46D | 1000 | **0.0792** | 3.0502 ❌ | **0.88%/dim** |
| 8a | Prosodic | 13D | 1000 | 0.1233 | - | 0.56%/dim |
| 8b | Spectral | 30D | 1000 | 0.1228 | - | 0.26%/dim |
| 8c | No Prosodic | 33D | 1000 | 0.0988 | - | 0.78%/dim |
| 9a | Acoustic | 46D | 100 (test) | - | 3.0502 | - |
| 10a (TODO) | Acoustic Norm | 46D | 1000 | ~0.08 | **<0.5** 🎯 | ~0.85%/dim |
| 10b (TODO) | Relative | ~40D | 1000 | ~0.09 | **<1.0** 🎯 | ~0.75%/dim |
| 11 (TODO) | Acoustic+Temporal | ~74D | 1000 | ~0.07 | **<0.4** 🎯 | ~1.0%/dim |
| 12 (TODO) | Multi-Speaker | 46D | 1000 (4×250) | ~0.10 | **<0.3** 🎯 | ~0.65%/dim |

## 🔧 待实现功能清单

### 高优先级 (本周)
- [ ] `scripts/compute_speaker_statistics.py` - 计算per-speaker统计量
- [ ] `audio_augmented_llm/src/emotion_encoder/normalized_features.py` - 归一化特征提取
- [ ] `scripts/extract_normalized_features.py` - 批量归一化
- [ ] Experiment 10a执行 + 评估
- [ ] Experiment 10b relative features实现

### 中优先级 (下周)
- [ ] `audio_augmented_llm/src/emotion_encoder/contour_features.py` - DCT轮廓
- [ ] Multi-scale variance features
- [ ] Experiment 11执行
- [ ] 对比temporal vs static features

### 低优先级 (后续)
- [ ] `scripts/generate_multispeaker_dataset.py` - 多speaker数据生成
- [ ] Experiment 12执行
- [ ] Hybrid模型 (normalized + WavLM)
- [ ] IEMOCAP/RAVDESS真实语音测试

### 论文完善
- [ ] Related Work: 补充20+篇引用
- [ ] Discussion: 分析generalization failure原因
- [ ] Figures: 架构图、scaling曲线、ablation plots
- [ ] References.bib: 整理文献

## 📈 项目里程碑

### ✅ 已完成 (Phase 1-3)
- [x] Phase 1: Proof of Concept (Exp 1-3)
- [x] Phase 2: Scale Validation (Exp 4-6)
- [x] Phase 3: Feature Engineering (Exp 7-8)
- [x] Generalization Testing (Exp 9a)
- [x] Paper Framework建立
- [x] 46D acoustic features设计与实现
- [x] Feature ablation analysis
- [x] Negative result documentation

### 🔄 进行中 (Phase 4)
- [ ] Speaker Normalization (Exp 10a-b)
- [ ] Temporal Dynamics (Exp 11)
- [ ] Multi-Speaker Training (Exp 12)

### ⏳ 待启动 (Phase 5)
- [ ] Real Human Speech Testing
- [ ] Production Optimization
- [ ] Paper Polishing
- [ ] Submission Preparation

## 💡 关键洞察与教训

### ✅ 成功经验
1. **Speech science原理有效**: 基于物理模型的特征设计work
2. **Feature synergy存在**: 组合特征 > 单独特征之和
3. **维度效率重要**: 46D vs 256D，性能更好且更高效
4. **Ablation研究价值**: 帮助理解哪些特征最重要

### ⚠️ 失败教训
1. **Absolute features不泛化**: Speaker差异dominant
2. **Static aggregation丢失信息**: 时序动态critical
3. **Single-speaker training脆弱**: 需要多样化训练数据
4. **过早乐观**: In-domain性能 ≠ 泛化能力

### 🎓 理论贡献
1. **Negative results有价值**: 识别fundamental limitation
2. **Feature engineering需谨慎**: 不是所有hand-crafted features都generalize
3. **Normalization is critical**: 对acoustic features尤其重要
4. **Multi-scale analysis必要**: 情感存在于多个时间尺度

## 📝 数据资产清单

### 音频数据 (不在git，总计~15GB)
```
audio_augmented_llm/data/
├── train_100/         # 100样本 (早期实验)
├── train_500/         # 500样本
├── train_1000/        # 1000样本 (主训练集)
│   └── audio/         # 1000 × ~100KB = 100MB
├── test_cross_speaker_damien/  # 100样本
│   └── audio/         # 100 × ~100KB = 10MB
└── literary_texts/    # 442故事片段
```

### 特征embeddings (~500MB)
```
train_1000/
├── acoustic_embeddings.npz      # 46D × 1000 = 184KB
├── prosodic_only.npz            # 13D × 1000
├── spectral_only.npz            # 30D × 1000
├── no_prosodic.npz              # 33D × 1000
└── wavlm_embeddings.npz         # 256D × 1000 = 1MB

test_cross_speaker_damien/
└── acoustic_embeddings.npz      # 46D × 100
```

### 模型checkpoints (~2GB)
```
models/
├── exp7b_acoustic_1000/         # ~500MB (LoRA + projection)
├── exp8_prosodic_only/          # ~500MB
├── exp8_spectral_only/          # ~500MB
└── exp8_no_prosodic/            # ~500MB
```

## 🚀 下周行动计划

### Day 1-2 (Mon-Tue): Exp 10a Implementation
- 实现speaker statistics计算
- 实现normalized feature extraction
- 收集Claribel和Damien的statistics

### Day 3 (Wed): Exp 10a Training
- 提取训练集normalized features
- 训练模型 (5 epochs)
- 监控in-domain性能

### Day 4 (Thu): Exp 10a Evaluation
- 提取测试集normalized features
- 评估cross-speaker性能
- 对比Exp 9a结果

### Day 5 (Fri): Exp 10b Relative Features
- 实现relative feature extraction
- 训练和评估
- 写weekly report

### Weekend: Analysis & Documentation
- 分析Exp 10结果
- 更新paper Analysis section
- 准备Exp 11 implementation

## 📧 项目联系方式

**Researcher**: GeoffreyWang1117
**Email**: 173976389+GeoffreyWang1117@users.noreply.github.com
**GitHub**: https://github.com/GeoffreyWang1117/AudioLLMProject
**Branch**: alpha (experimental)

---

**文档生成时间**: 2025-11-15 22:45 UTC
**下次更新**: 完成Exp 10后
**项目状态**: Phase 4 in progress - 75% complete toward first submission
