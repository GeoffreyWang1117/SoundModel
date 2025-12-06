# Audio-Augmented Small LLM for Emotion- and Trend-Aware Reasoning

Research project exploring emotion-augmented knowledge distillation for small language models using acoustic features from synthesized speech.

**Target Conference**: ACL/ICML 2026
**Status**: Phase 4 - Feature Engineering & Generalization Analysis

## 🎯 Project Goal

Train small LLMs (1.5B parameters) that leverage emotion embeddings extracted from TTS-synthesized audio to improve reasoning capabilities, with focus on:
1. Speech science-based acoustic features (prosody, spectral, formants)
2. Computational efficiency (46D vs 256D embeddings)
3. Cross-speaker generalization

## 🔥 Latest Results (Updated 2025-11-15)

### Experiment 7b: Acoustic Features Breakthrough (1000 samples)
| Method | Features | Val Loss | Improvement | Efficiency |
|--------|----------|----------|-------------|------------|
| Text-Only | - | 0.1330 | - | - |
| WavLM | 256D | 0.0960 | 27.82% | 0.109%/dim |
| **Acoustic** | **46D** | **0.0792** | **40.45%** ⭐ | **0.88%/dim** |

**Key Achievement**: Acoustic features achieve **17.5% better performance** than WavLM with only **18% of dimensions** → **8.06× dimensional efficiency**

### Experiment 8: Feature Ablation Studies

| Feature Set | Dimensions | Val Loss | Improvement |
|-------------|-----------|----------|-------------|
| Full Acoustic | 46D | **0.0792** | **40.45%** |
| Spectral only | 30D | 0.1228 | 7.67% |
| Prosodic only | 13D | 0.1233 | 7.29% |
| w/o Prosodic | 33D | 0.0988 | 25.71% |

**Discovery**: **Feature Synergy Effect**
- Spectral features provide backbone (25.71% improvement)
- Adding prosodic features yields 40.45% (not 33.00%)
- **+7.45% synergy bonus** from prosodic-spectral interaction

### Experiment 9a: Cross-Speaker Generalization ⚠️

| Test Set | Speaker | Gender | Val Loss | Change |
|----------|---------|--------|----------|--------|
| In-domain | Claribel Dervla | Female | **0.0792** | - |
| Cross-speaker | Damien Black | Male | **3.0502** | **+3752%** |

**Critical Finding**: **Catastrophic generalization failure** (38.5× performance degradation)
- Reveals speaker-specific overfitting
- Challenges hypothesis that hand-crafted features generalize better
- Motivates speaker normalization and relative features

## 📊 Acoustic Feature Engineering

### Feature Design (46D Total)

#### 1. Prosodic Features (13D)
**Physical Basis**: F0 (fundamental frequency) via YIN algorithm
```python
Pitch features (7D):
- f0_mean, f0_std       # Central tendency
- f0_min, f0_max        # Range extremes
- f0_range              # Dynamic range
- f0_slope              # Temporal trend (linear regression)

Energy features (3D):
- rms_mean, rms_std     # Intensity
- rms_max               # Peak energy

Temporal features (3D):
- duration              # Total speech time
- speech_rate           # Voiced frames / duration
- zcr_mean, zcr_std     # Zero-crossing rate
```

**Mathematical Model**:
- YIN algorithm: Autocorrelation-based pitch detection with difference function
- Speech rate: `r = N_voiced / T` where T is duration
- ZCR: `zcr(t) = (1/2N) Σ|sign(x[n]) - sign(x[n-1])|`

#### 2. Spectral Features (30D)
**Physical Basis**: Mel-frequency cepstral coefficients (MFCCs)
```python
MFCCs (13D):          # Mel-scale spectral envelope
Delta MFCCs (13D):    # Temporal dynamics (1st derivative)
Spectral stats (4D):
- spectral_centroid   # Brightness (weighted mean frequency)
- spectral_rolloff    # High-frequency cutoff (85% energy)
- spectral_flux       # Frame-to-frame change
- spectral_bandwidth  # Frequency spread
```

**Mathematical Model**:
- MFCC: `X[k] = Σ log(S[m]) · cos(k(m - 0.5)π/M)` where S[m] is Mel-filterbank output
- Centroid: `C = Σ(f · X[f]) / Σ X[f]` (center of gravity)
- Delta: `Δ[t] = (X[t+1] - X[t-1]) / 2` (discrete derivative)

#### 3. Formant Features (3D)
**Physical Basis**: Vocal tract resonances via Linear Predictive Coding (LPC)
```python
F1, F2, F3  # First three formant frequencies
```

**Mathematical Model**:
- LPC: `x[n] = -Σ a[k]·x[n-k] + G·e[n]` (autoregressive model)
- Formants: Peaks in frequency response `H(z) = G / (1 + Σ a[k]z^-k)`

### Why These Features Failed Cross-Speaker

**Problem**: Absolute values are speaker-dependent
- Female F0: 200-400 Hz (Claribel Dervla)
- Male F0: 100-150 Hz (Damien Black)
- Model learned speaker identity, not emotion patterns

**Root Cause**:
1. **No normalization**: Features not standardized within speaker
2. **Absolute vs relative**: Used pitch_mean instead of pitch_slope
3. **Static aggregation**: Mean/std lose temporal dynamics

## 🔬 声学物理模型与数学原理

### 1. 语音产生的源-滤波器模型 (Source-Filter Model)

**物理模型**:
```
Speech = Excitation (声带振动) × Vocal Tract Filter (共振腔)
```

**数学表达**:
```
s(t) = e(t) * h(t)
S(ω) = E(ω) · H(ω)
```

其中：
- `e(t)`: 激励信号（声带振动，决定F0/pitch）
- `h(t)`: 声道滤波器（决定formants/音色）
- `*`: 卷积运算

**情感关联**:
- **Excitation (F0)**: 情绪影响声带张力 → 改变pitch
  - 兴奋：F0升高，range增大
  - 悲伤：F0降低，range减小
- **Filter (Formants)**: 情绪影响声道形状 → 改变共振
  - 紧张：喉部收缩 → F1降低
  - 放松：声道开放 → F1升高

### 2. 梅尔频率倒谱系数 (MFCC) 的心理声学基础

**物理-生理映射**:
```
物理频率 f (Hz) → 梅尔频率 m (Mel)
m = 2595 · log₁₀(1 + f/700)
```

**心理声学原理**:
- 人耳对低频敏感，对高频分辨率降低
- 梅尔刻度模拟人耳感知的非线性特性
- 临界频带 (Critical Bands): 200 Hz @ 低频，3500 Hz @ 高频

**情感编码**:
- MFCC低阶系数：谱包络 → 音色 → 情感基调
- MFCC高阶系数：谱细节 → 粗糙度 → 唤醒度
- Delta MFCC：时间动态 → 韵律变化 → 情感强度

### 3. 基频 (F0) 检测的YIN算法

**数学原理**:
```
差分函数: d_t(τ) = Σ(x_j - x_{j+τ})²
归一化: d'_t(τ) = d_t(τ) / [(1/τ)Σd_t(j)]
```

**物理意义**:
- 自相关的改进版本，避免能量偏差
- τ对应基频周期: F0 = f_s / τ
- 阈值法选择最佳周期估计

**情感维度**:
- F0_mean: 情感价态 (valence)
- F0_std: 情感唤醒度 (arousal)
- F0_slope: 情感动态变化

### 4. 线性预测编码 (LPC) 与共振峰提取

**数学模型**:
```
全极点模型: H(z) = G / (1 - Σa_k·z^(-k))
```

**物理解释**:
- 系数 a_k 描述声道滤波器特性
- 极点位置对应共振峰频率
- G 为增益，对应声门激励强度

**共振峰-情感映射**:
- F1-F2空间：元音三角形，情感改变发音
- F1高：张开度大 → 兴奋、惊讶
- F1低：张开度小 → 悲伤、恐惧

## 🔧 当前问题与改进方案

### Problem 1: Speaker-Dependent Features

**Current Approach** (失败):
```python
# 绝对值特征
features['pitch_mean'] = np.mean(f0)  # 200 Hz (female) vs 100 Hz (male)
```

**Improved Approach 1**: Speaker Normalization
```python
# Z-score归一化（需要speaker statistics）
f0_norm = (f0 - speaker_f0_mean) / speaker_f0_std
features['pitch_mean_norm'] = np.mean(f0_norm)  # 0 ± 1 for all speakers
```

**Improved Approach 2**: Relative Features
```python
# 相对变化而非绝对值
features['pitch_range_rel'] = (f0_max - f0_min) / f0_mean  # 归一化范围
features['pitch_slope'] = np.polyfit(time, f0, 1)[0]       # 趋势而非水平
```

**Improved Approach 3**: Percentile Features
```python
# 使用分位数而非绝对值
features['pitch_p10'] = np.percentile(f0, 10)
features['pitch_p50'] = np.percentile(f0, 50)  # median
features['pitch_p90'] = np.percentile(f0, 90)
# 然后计算相对距离
features['pitch_spread'] = (p90 - p10) / p50
```

### Problem 2: Static Aggregation Loses Temporal Dynamics

**Current Approach** (失败):
```python
# 丢失时间信息
pitch_mean = np.mean(f0)  # 单一标量
```

**Improved Approach**: Temporal Statistics
```python
# 保留时序动态
def extract_temporal_stats(signal):
    # 分段统计（早期/中期/晚期）
    early = signal[:len(signal)//3]
    mid = signal[len(signal)//3:2*len(signal)//3]
    late = signal[2*len(signal)//3:]

    features = {
        'early_mean': np.mean(early),
        'mid_mean': np.mean(mid),
        'late_mean': np.mean(late),
        'trend': (np.mean(late) - np.mean(early)) / np.mean(signal),  # 归一化趋势
    }
    return features
```

**Improved Approach 2**: Contour Modeling
```python
# 韵律轮廓拟合
def extract_pitch_contour(f0, n_coeffs=5):
    # 离散余弦变换 (DCT) 压缩轮廓
    from scipy.fftpack import dct
    time = np.arange(len(f0))

    # Normalize time and pitch
    f0_norm = (f0 - np.mean(f0)) / np.std(f0)

    # DCT coefficients capture contour shape
    dct_coeffs = dct(f0_norm, norm='ortho')

    return dct_coeffs[:n_coeffs]  # Low-order coeffs = global shape
```

### Problem 3: Missing Multi-Scale Analysis

**Improved Approach**: Multi-Resolution Features
```python
def extract_multiscale_features(signal, sr=16000):
    features = {}

    # Short-term (phone level, ~50ms)
    hop_short = int(0.05 * sr)
    features['short_term_variance'] = np.var([
        np.mean(signal[i:i+hop_short])
        for i in range(0, len(signal), hop_short)
    ])

    # Medium-term (syllable level, ~200ms)
    hop_med = int(0.2 * sr)
    features['med_term_variance'] = np.var([
        np.mean(signal[i:i+hop_med])
        for i in range(0, len(signal), hop_med)
    ])

    # Long-term (utterance level)
    features['long_term_mean'] = np.mean(signal)

    return features
```

## 🎯 下一步实验计划

### Phase 4A: Speaker-Normalized Features (优先级：高)

**Experiment 10a**: Z-Score Normalization
```bash
# 实现speaker-level归一化
python scripts/extract_acoustic_features_normalized.py \
  --normalization z-score \
  --speaker_stats speaker_statistics.json
```

**Expected Result**: 显著提升cross-speaker性能 (目标: 3.0502 → <0.5)

**Experiment 10b**: Relative Features
```bash
# 使用相对特征而非绝对值
python scripts/extract_relative_features.py \
  --feature_mode relative \
  --percentile_based
```

**Hypothesis**: 相对特征capture emotion-specific patterns独立于speaker identity

### Phase 4B: Temporal Dynamics (优先级：高)

**Experiment 11**: Contour-Based Features
```bash
# DCT轮廓系数 + 分段统计
python scripts/extract_contour_features.py \
  --method dct \
  --n_coeffs 8 \
  --temporal_segments 3
```

**Expected Dimension**: ~60D (DCT 8 + temporal stats 20 + existing 30)

### Phase 4C: Multi-Speaker Training (优先级：中)

**Experiment 12**: Diverse TTS Speakers
```bash
# 使用多个speaker训练
python scripts/generate_multispeaker_dataset.py \
  --speakers "Claribel Dervla,Damien Black,Gilberto Mathias,Royston Min" \
  --samples_per_speaker 250  # 4 speakers × 250 = 1000
```

**Hypothesis**: 强制模型学习speaker-invariant emotion patterns

### Phase 4D: Hybrid Approach (优先级：中)

**Experiment 13**: Normalized Acoustics + WavLM
```bash
# 结合归一化acoustic features和WavLM
python scripts/train_student_simple.py \
  --embedding_type fusion \
  --acoustic_normalized \
  --acoustic_dim 46 \
  --wavlm_dim 256
```

**Expected Performance**: Best of both worlds (interpretability + learned patterns)

### Phase 5: Real Human Speech Testing (优先级：低)

**Experiment 14**: IEMOCAP/RAVDESS Transfer
```bash
# 测试TTS → Real speech generalization
python scripts/evaluate_on_iemocap.py \
  --model best_acoustic_normalized \
  --dataset IEMOCAP
```

## 📂 Updated Project Structure

```
.
├── audio_augmented_llm/
│   ├── src/
│   │   ├── tts_pipeline/              # XTTS v2 synthesis
│   │   ├── emotion_encoder/
│   │   │   ├── acoustic_features.py   # Current 46D features
│   │   │   ├── normalized_features.py # TODO: Speaker normalization
│   │   │   ├── contour_features.py    # TODO: Temporal dynamics
│   │   │   └── wavlm_encoder.py       # WavLM baseline
│   │   └── student_training/          # QLoRA training
│   ├── data/
│   │   ├── train_1000/                # Main training set
│   │   ├── test_cross_speaker_damien/ # Generalization test
│   │   └── speaker_statistics/        # TODO: For normalization
│   └── models/
│       ├── exp7b_acoustic_1000/       # Best model (val: 0.0792)
│       └── exp8_*/                    # Ablation checkpoints
├── scripts/
│   ├── generate_*.py                  # Data generation
│   ├── extract_*.py                   # Feature extraction
│   ├── train_*.py                     # Model training
│   ├── evaluate_*.py                  # Evaluation
│   └── run_*.sh                       # Batch runners
├── paper/                             # LaTeX manuscript
│   ├── main.tex
│   └── sections/
│       ├── abstract.tex
│       ├── introduction.tex
│       ├── methodology.tex
│       ├── experimental_setup.tex
│       ├── results.tex
│       ├── analysis.tex              # Updated with Exp 8, 9a
│       ├── discussion.tex            # TODO
│       └── conclusion.tex
├── outputs/                           # Training logs
├── EXPERIMENT_LOG.md                  # Detailed records
├── EXPERIMENT_9_RESULTS.md            # Generalization analysis
└── README.md                          # This file
```

## 🛠️ Technical Stack

- **Base Model**: Qwen2.5-1.5B (4-bit quantization + LoRA r=16, α=32)
- **TTS**: XTTS v2 (Coqui, multi-lingual, speaker: Claribel Dervla)
- **Feature Extraction**:
  - Librosa 0.10.1 (MFCC, spectral features)
  - Custom YIN implementation (F0 detection)
  - Praat-inspired LPC (formant extraction)
- **Training**: PyTorch 2.5.1, Transformers 4.45.2, PEFT 0.13.2
- **GPU**: NVIDIA (22GB VRAM available)

## 📊 Complete Results Summary

| Exp | Method | Features | Samples | Val Loss | vs Baseline |
|-----|--------|----------|---------|----------|-------------|
| 5 | WavLM | 256D | 1000 | 0.0960 | +27.82% |
| 7b | Acoustic | 46D | 1000 | **0.0792** | **+40.45%** |
| 8a | Prosodic only | 13D | 1000 | 0.1233 | +7.29% |
| 8b | Spectral only | 30D | 1000 | 0.1228 | +7.67% |
| 8c | w/o Prosodic | 33D | 1000 | 0.0988 | +25.71% |
| 9a | Cross-speaker | 46D | 100 | 3.0502 | **-3752%** ⚠️ |

**Key Insights**:
- ✅ Acoustic features achieve best in-domain performance
- ✅ Feature synergy discovered (+7.45% bonus)
- ⚠️ Catastrophic cross-speaker failure reveals normalization need
- 🎯 Next: Speaker normalization + temporal dynamics

## 🎓 Research Contributions

1. **Dimensional efficiency**: 8.06× better performance per dimension than WavLM
2. **Feature synergy discovery**: Non-additive interaction between prosodic and spectral
3. **Generalization analysis**: Identified speaker-dependency as critical limitation
4. **Negative results**: Hand-crafted features don't automatically generalize better
5. **Clear path forward**: Speaker normalization + relative features + temporal dynamics

## 🚀 Quick Start

```bash
# 1. Setup environment
conda create -n audio_llm python=3.10
conda activate audio_llm
pip install torch transformers trl peft TTS librosa scipy

# 2. Generate synthetic data
python scripts/generate_teacher_data.py --num_samples 100

# 3. Extract acoustic features
python scripts/extract_acoustic_features_batch.py \
  --data_dir ./audio_augmented_llm/data/train_100

# 4. Train model with acoustic features
python scripts/train_student_simple.py \
  --data_dir ./audio_augmented_llm/data/train_100 \
  --embedding_type acoustic \
  --use_emotion \
  --num_epochs 5 \
  --batch_size 4

# 5. Evaluate on cross-speaker test
python scripts/evaluate_model.py \
  --model_dir ./audio_augmented_llm/models/best_model \
  --data_dir ./audio_augmented_llm/data/test_cross_speaker_damien \
  --embedding_type acoustic
```

## 📧 Contact

GeoffreyWang1117
Email: 173976389+GeoffreyWang1117@users.noreply.github.com
GitHub: https://github.com/GeoffreyWang1117/AudioLLMProject

## 📄 License

Research project - License TBD

---

**Last Updated**: 2025-11-15 22:30 UTC
**Branch**: alpha (experimental development)
**Next Milestone**: Speaker-normalized features (Exp 10)
**Paper Submission Target**: ACL/ICML 2026 (submission ~2025-12 or 2026-01)
