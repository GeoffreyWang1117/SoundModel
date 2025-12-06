# 快速参考命令

## 🔍 检查Exp 13数据生成进度

### 查看实时日志
```bash
# 4说话人训练数据生成进度
tail -f outputs/exp13_4speaker_generation_log.txt

# Viktor测试集生成进度  
tail -f outputs/exp13_viktor_test_generation_log.txt
```

### 检查进程状态
```bash
# 查看正在运行的生成进程
ps aux | grep "python.*generate.*speaker" | grep -v grep
```

### 检查已生成的数据
```bash
# 查看训练数据目录
ls -lh audio_augmented_llm/data/exp13_4speaker/train/

# 查看验证数据目录
ls -lh audio_augmented_llm/data/exp13_4speaker/val/

# 查看Viktor测试数据
ls -lh audio_augmented_llm/data/test_cross_speaker_viktor/

# 统计已生成的音频文件数
find audio_augmented_llm/data/exp13_4speaker/train/audio/ -name "*.wav" | wc -l
find audio_augmented_llm/data/exp13_4speaker/val/audio/ -name "*.wav" | wc -l
find audio_augmented_llm/data/test_cross_speaker_viktor/audio/ -name "*.wav" | wc -l
```

### 预期完成数量
- 训练集: 800 音频文件 (4 speakers × 200 each)
- 验证集: 80 音频文件 (4 speakers × 20 each)  
- Viktor测试集: 100 音频文件

---

## 🚀 数据生成完成后的下一步

### 1. 训练Exp 13模型

```bash
source /home/coder-gw/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm

python scripts/train_student_simple.py \
  --train_dir ./audio_augmented_llm/data/exp13_4speaker/train \
  --val_dir ./audio_augmented_llm/data/exp13_4speaker/val \
  --embedding_type wavlm \
  --emotion_dim 256 \
  --num_epochs 5 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --output_dir ./audio_augmented_llm/models/exp13_4speaker \
  2>&1 | tee outputs/exp13_training_log.txt
```

### 2. 评估模型

```bash
# 验证集评估（同说话人）
python scripts/evaluate_model.py \
  --model_dir ./audio_augmented_llm/models/exp13_4speaker/best_model \
  --data_dir ./audio_augmented_llm/data/exp13_4speaker/val \
  --embedding_type wavlm \
  --batch_size 8

# 跨说话人评估（Viktor - 完全未见）
python scripts/evaluate_model.py \
  --model_dir ./audio_augmented_llm/models/exp13_4speaker/best_model \
  --data_dir ./audio_augmented_llm/data/test_cross_speaker_viktor \
  --embedding_type wavlm \
  --batch_size 8
```

---

## 📊 查看实验结果

```bash
# 查看完整实验日志
cat EXPERIMENT_LOG.md

# 查看Exp 12a结果
cat EXPERIMENT_12A_RESULTS.md

# 查看Exp 11结果
cat EXPERIMENT_11_RESULTS.md

# 查看跨说话人分析
cat CROSS_SPEAKER_ANALYSIS.md
```

---

## 📄 查看论文

```bash
# 查看PDF（如果有PDF阅读器）
evince paper/main.pdf &
# 或
xdg-open paper/main.pdf &

# 查看LaTeX源文件
cat paper/main.tex
cat paper/sections/5_results.tex  # 查看结果章节
cat paper/sections/6_discussion.tex  # 查看讨论章节
```

---

## 🔬 Future Work参考

```bash
# 查看完整的未来工作计划
cat FUTURE_WORK.md

# 查看下一步实验计划
cat NEXT_EXPERIMENTS_PLAN.md

# 查看今日会话总结
cat SESSION_SUMMARY_2025-11-17.md
```

---

## 🛠 常用调试命令

### 检查GPU使用
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 每秒更新
```

### 检查磁盘空间
```bash
df -h .
du -sh audio_augmented_llm/data/*
```

### 检查模型文件
```bash
# 列出所有训练好的模型
ls -lh audio_augmented_llm/models/

# 查看特定模型的配置
cat audio_augmented_llm/models/exp11_2speaker/best_model/training_config.json
```

### 清理旧的后台进程
```bash
# 如果需要停止数据生成
pkill -f "python.*generate.*speaker"

# 查看所有Python进程
ps aux | grep python
```

---

## 📈 性能基准

### 已知结果（用于对比）

| 实验 | 训练 | 测试 | Loss | 说明 |
|------|------|------|------|------|
| Exp 7b | Claribel (1000) | Claribel val | 0.0792 | 域内基准 |
| Exp 9a | Claribel (1000) | Damien | 3.0502 | 单说话人跨测试 |
| Exp 11 | Claribel+Damien (550) | Damien val | 0.2468 | 同说话人（误导） |
| Exp 11 | Claribel+Damien (550) | Andrew | 2.5503 | 真实跨说话人 |
| Exp 12a | Claribel (1000) acoustic | Damien | 3.2264 | Acoustic跨测试 |

### Exp 13目标

| 指标 | 目标值 | 理由 |
|------|--------|------|
| 验证集loss | < 0.3 | 同说话人性能 |
| Viktor测试loss | < 2.0 | 比Exp 11(2.5503)改善20%+ |
| 改善程度 | > 20% | 证明4说话人有效 |

---

## 🆘 遇到问题？

### 导入错误
```bash
# 确保在项目根目录
cd /home/coder-gw/Projects/Emotional-Aware-Reasoning

# 激活环境
source /home/coder-gw/miniconda3/etc/profile.d/conda.sh
conda activate audio_llm

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 内存不足
```bash
# 减小batch size
--batch_size 2  # 而不是4

# 减少样本数
--samples_per_speaker 150  # 而不是200
```

### CUDA错误
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 查看GPU状态
nvidia-smi
```

---

**最后更新**: 2025-11-17
**当前阶段**: Exp 13 数据生成中
**预计完成**: 4-5小时
