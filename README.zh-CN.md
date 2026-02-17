# RVC_WAV_Effective（中文说明）

一个实用的 Python 工具，用于分析 RVC（Retrieval-based Voice Conversion）训练中 WAV 文件的**有效人声时长**。

---

本工具面向噪声较低、录音室级别的干净音频，帮助你：

- 估算可用人声总时长
- 降低碎片化（过短断裂片段）
- 保留气声与细微发声细节
- 导出结构化分段数据，便于数据集制作

---

## 为什么要用这个工具？

在为 RVC 准备数据集时，你通常会遇到：

- 朴素 VAD 导致分段过碎
- 气声被错误移除
- 极低噪声地板导致自动阈值估计不稳定
- 难以评估“真正可用”的语音数据量

`rvc_wav_effective` 通过以下方法解决这些问题：

- 基于能量的帧级分析（RMS dBFS）
- 使用百分位参考的智能自动阈值
- 带迟滞（hysteresis）的 VAD（减少微碎片）
- 静音填充与片段合并
- 可选边界 padding，保留起音/尾音/气声
- 结构化 JSON 导出

---

## 功能特性

- 帧级 RMS 能量分析
- 稳定的自动阈值估计（对无噪声录音更稳健）
- 迟滞 VAD（开启/关闭双阈值）
- 静音桥接
- 短片段过滤
- 片段边界扩展（padding）
- 面向数据集流水线的 JSON 导出
- 所有参数均可通过 CLI 配置

---

## 安装

需要 Python 3.8+

```bash
pip install numpy soundfile
```

克隆仓库：

```bash
git clone https://github.com/yourname/rvc_wav_effective.git
cd rvc_wav_effective
```

---

## 基础用法

```bash
python rvc_wav_effective.py input.wav
```

---

## 录音室 RVC 数据集推荐参数

```bash
python rvc_wav_effective.py FireFly.wav \
  --thr_mode auto \
  --relative_drop_db 34 \
  --hyst_db 4 \
  --min_voice_ms 450 \
  --fill_sil_ms 500 \
  --merge_gap_ms 700 \
  --pad_ms 120 \
  --dump_json segments.json
```

该配置可以：

- 保留气声
- 减少过短碎片段
- 产生更接近完整句子的分段
- 适用于 RVC 训练数据集

---

## 示例输出

```
==== RVC Effective Vocal Duration Analysis ====
File: FireFly.wav
Sample rate: 48000 Hz
Total duration: 1378.069 s

Auto Threshold Info:
  Reference percentile: P90 = -19.01 dBFS
  Threshold (thr_on): -53.01 dBFS
  Hysteresis off threshold: -57.01 dBFS

Effective segments: 254
Effective vocal duration: 818.440 s (59.4%)

Segment statistics:
  Min: 0.720 s
  Max: 12.090 s
  Median: 2.665 s
  <1.0s segments: 25 / 254
```

---

## 自动阈值原理

与其从低分位数估计噪声地板（这种方式在数字化干净录音中容易失效），
本工具采用：

```
threshold = P(ref_percentile) - relative_drop_db
```

其中：

- `P(ref_percentile)` 表示高能量参考水平（默认：第 90 百分位）
- `relative_drop_db` 控制比语音能量低多少仍判定为有效
- 阈值会被限制在可配置范围内，防止不稳定

这种方法尤其适用于：

- 录音室音频
- 低噪声语音
- 需要保留气声的数据集

---

## 关键参数说明

### 阈值控制

| 参数 | 说明 |
|------------|------------|
| `--thr_mode` | `auto` 或 `manual` |
| `--relative_drop_db` | 越小越容易保留气声 |
| `--hyst_db` | 迟滞区间（可降低碎片化） |
| `--clamp_min_db` | 防止阈值过低 |

---

### 碎片化控制

| 参数 | 说明 |
|------------|------------|
| `--min_voice_ms` | 片段最小时长 |
| `--fill_sil_ms` | 填补句内短静音 |
| `--merge_gap_ms` | 合并相近片段 |
| `--pad_ms` | 扩展片段边界 |

---

## 常见调参配置

### 保留气声、减少碎片（推荐）

```
relative_drop_db: 32~34
hyst_db: 3~4
min_voice_ms: 400~500
fill_sil_ms: 400~600
merge_gap_ms: 600~900
pad_ms: 100~150
```

### 严格模式（仅语音）

```
relative_drop_db: 36~40
min_voice_ms: 600+
```

---

## JSON 输出格式

```json
{
  "wav": "FireFly.wav",
  "sr": 48000,
  "duration_s": 1378.069,
  "thr_on_db": -53.01,
  "thr_off_db": -57.01,
  "effective_total_s": 818.440,
  "segments": [
    {
      "start_s": 12.340,
      "end_s": 15.870,
      "dur_s": 3.530
    }
  ]
}
```

---

## 适用场景

- RVC 数据集制作
- 数据集时长校验
- 保留气声的人声建模
- 自动预分段
- 训练数据质量诊断

附：RVC轮数建议表

| 数据质量 \ 有效人声时长 | 短：1–3 分钟 | 中：3–10 分钟 | 长：10–30 分钟 |
|:--|:--:|:--:|:--:|
| 好（棚录/干声/无损/少处理） | 80–200E | 150–350E | 200–450E |
| 一般（轻压缩/轻降噪/轻混响或轻去齿） | 60–160E | 120–300E | 150–400E |
| 差（明显压缩伪影/混响重/降噪重/噪声大） | 40–120E | 80–220E | 100–280E |

---

## 局限性

- 仅基于能量（不做音素级检测）
- 不适合高噪声环境
- 假设录音本身较干净

---

## 未来改进方向

- 面向 RVC 分块（如 3–6 秒）的目标长度切分
- 可视化模式
- 可选频谱特征检测
- 多文件批处理

---

## 许可证

GPL v3.0

---

## 贡献

欢迎提交 PR 与功能建议。

如果这个工具提升了你的 RVC 工作流，欢迎点个 Star。
