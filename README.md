# RVC_WAV_Effective

A practical Python tool to analyze **effective vocal duration** in WAV files for RVC (Retrieval-based Voice Conversion) training.

## Language / 语言

- English: [README.md](README.md)
- 简体中文: [README.zh-CN.md](README.zh-CN.md)

---

Designed for clean studio recordings with minimal noise, this tool helps you:

- Estimate usable vocal duration
- Reduce fragmentation (short broken segments)
- Preserve breath and subtle vocal details
- Export structured segment data for dataset preparation

---

## Why This Tool?

When preparing datasets for RVC, you often face:

- Over-fragmented segments caused by naive VAD
- Breath sounds being incorrectly removed
- Extremely low noise floor causing unstable auto-threshold estimation
- Difficulty estimating how much *actually usable* data you have

`rvc_wav_effective` solves these issues with:

- Energy-based frame analysis (RMS dBFS)
- Smart auto-thresholding using percentile reference
- Hysteresis-based VAD (reduces micro-fragmentation)
- Silence filling and segment merging
- Optional padding to preserve attack/tail/breath
- Structured JSON export

---

## Features

- Frame-level RMS energy analysis
- Stable auto-threshold estimation (robust for noise-free recordings)
- Hysteresis VAD (open/close thresholds)
- Silence bridging
- Short-segment filtering
- Segment padding
- JSON export for dataset pipelines
- Fully parameterized via CLI

---

## Installation

Requires Python 3.8+

```bash
pip install numpy soundfile
```

Clone the repository:

```bash
git clone https://github.com/yourname/rvc_wav_effective.git
cd rvc_wav_effective
```

---

## Basic Usage

```bash
python rvc_wav_effective.py input.wav
```

---

## Recommended Usage for Studio RVC Dataset

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

This configuration:

- Preserves breath sounds
- Reduces short fragment segments
- Produces more sentence-like segments
- Suitable for RVC training datasets

---

## Example Output

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

## How Auto Threshold Works

Instead of estimating noise floor from low percentiles (which fails for digitally clean recordings),
this tool uses:

```
threshold = P(ref_percentile) - relative_drop_db
```

Where:

- `P(ref_percentile)` represents a high-energy reference level (default: 90th percentile)
- `relative_drop_db` controls how much weaker than speech energy is still considered valid
- Threshold is clamped between configurable limits to prevent instability

This approach works well for:

- Studio recordings
- Noise-free speech
- Breath-preserving datasets

---

## Key Parameters Explained

### Threshold Control

| Parameter | Description |
|------------|------------|
| `--thr_mode` | `auto` or `manual` |
| `--relative_drop_db` | Lower = more breath preserved |
| `--hyst_db` | Hysteresis gap (reduces fragmentation) |
| `--clamp_min_db` | Prevent threshold from being too low |

---

### Fragmentation Control

| Parameter | Description |
|------------|------------|
| `--min_voice_ms` | Minimum segment duration |
| `--fill_sil_ms` | Fill short silences inside a sentence |
| `--merge_gap_ms` | Merge nearby segments |
| `--pad_ms` | Expand segment boundaries |

---

## Typical Tuning Profiles

### Preserve Breath, Reduce Fragmentation (Recommended)

```
relative_drop_db: 32~34
hyst_db: 3~4
min_voice_ms: 400~500
fill_sil_ms: 400~600
merge_gap_ms: 600~900
pad_ms: 100~150
```

### Strict (Speech Only)

```
relative_drop_db: 36~40
min_voice_ms: 600+
```

---

## JSON Output Format

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

## Intended Use Cases

- RVC dataset preparation
- Dataset duration validation
- Breath-preserving voice modeling
- Automatic pre-segmentation
- Training data quality diagnostics

---

## Limitations

- Energy-based only (no phoneme-level detection)
- Not designed for heavily noisy environments
- Assumes reasonably clean recordings

---

## Future Improvements

- Target-length slicing for RVC chunking (e.g., 3–6 seconds)
- Visualization mode
- Optional spectral-based detection
- Multi-file batch processing

---

## License

GPL v3.0

---

## Contributing

Pull requests and feature suggestions are welcome.

If this tool improves your RVC workflow, feel free to star the repository.
