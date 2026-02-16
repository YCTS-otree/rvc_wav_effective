import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf


# -----------------------------
# 数据结构：一个人声片段（秒）
# -----------------------------
@dataclass
class Segment:
    start_s: float
    end_s: float

    @property
    def dur_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


# -----------------------------
# 工具：转单声道
# -----------------------------
def to_mono(x: np.ndarray) -> np.ndarray:
    # 如果本来就是一维，说明已经是 mono
    if x.ndim == 1:
        return x
    # 多声道则按通道平均，避免只取某一路导致偏差
    return np.mean(x, axis=1)


# -----------------------------
# 工具：把音频分帧并计算 RMS(dBFS)
# -----------------------------
def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    # 把毫秒换算为采样点数
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)

    # 防呆：避免参数过小导致帧长为 0
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_ms/hop_ms 太小，导致 frame_len 或 hop_len 为 0")

    n = len(x)

    # 如果音频比一帧还短，就按单帧处理
    if n < frame_len:
        # 用 float64 计算更稳
        rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)
        # 以满幅 1.0 为参考，所以这是 dBFS
        db = 20.0 * np.log10(rms + 1e-12)
        return np.array([0.0]), np.array([db])

    # 可计算的总帧数（只取完整帧，稳定优先）
    num_frames = 1 + (n - frame_len) // hop_len

    # 预分配数组，避免循环里反复扩容
    rms = np.empty(num_frames, dtype=np.float64)
    times = np.empty(num_frames, dtype=np.float64)

    # 逐帧计算 RMS
    for i in range(num_frames):
        # 当前帧起点（采样点索引）
        start = i * hop_len
        # 截取一帧并转 float64
        frame = x[start:start + frame_len].astype(np.float64)
        # RMS = sqrt(mean(x^2))
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
        # 记录该帧的时间戳（秒）
        times[i] = start / sr

    # 转换到 dBFS
    rms_db = 20.0 * np.log10(rms + 1e-12)
    return times, rms_db


# -----------------------------
# 更稳的 auto 阈值估计（适合“底噪≈0”的音频）
# -----------------------------
def estimate_threshold_db(
    rms_db: np.ndarray,
    mode: str,
    manual_thr_db: float,
    # 下面这些只在 auto 模式生效
    ref_percentile: float = 90.0,
    relative_drop_db: float = 35.0,
    clamp_min_db: float = -70.0,
    clamp_max_db: float = -20.0,
) -> Tuple[float, Optional[dict]]:
    """
    你的音频“数字静音很多”，用低分位数估噪声底会得到 -99dBFS 这种不实用结果。
    所以这里改用：阈值 = P(ref) - relative_drop_db，并夹在 [clamp_min_db, clamp_max_db] 内。

    - P(ref)（比如 90 分位）大致代表“讲话时的能量尺度”
    - relative_drop_db 控制你愿意把多轻的气声/尾音算作有效
    - clamp_min_db 防止阈值跌到地心（比如 -87）
    - clamp_max_db 防止阈值过高把轻声掐死
    """
    if mode == "manual":
        return manual_thr_db, None

    # 选一个较高分位数当“人声能量参考”
    ref_db = float(np.percentile(rms_db, ref_percentile))

    # 用“相对下降量”得到阈值（越大越严格，越小越宽松）
    thr_db = ref_db - relative_drop_db

    # 给阈值加上下限，避免数字静音导致 auto 阈值离谱
    thr_db = max(thr_db, clamp_min_db)

    # 给阈值加上上限，避免 ref_db 本身太小导致阈值太高（少见但防一手）
    thr_db = min(thr_db, clamp_max_db)

    dbg = {
        "ref_percentile": ref_percentile,
        "ref_db": ref_db,
        "relative_drop_db": relative_drop_db,
        "clamp_min_db": clamp_min_db,
        "clamp_max_db": clamp_max_db,
    }
    return float(thr_db), dbg


# -----------------------------
# 语音活动检测（带滞回），尽量减少碎片
# -----------------------------
def vad_hysteresis(
    rms_db: np.ndarray,
    thr_on_db: float,
    thr_off_db: float,
) -> np.ndarray:
    """
    滞回策略：
    - 能量上升到 thr_on_db 才进入 voiced
    - 降到 thr_off_db 才退出 voiced
    这样不会因为能量在阈值附近抖动导致碎片化。
    """
    voiced = np.zeros_like(rms_db, dtype=bool)

    # 标记当前是否处于“人声段”
    in_seg = False

    for i, db in enumerate(rms_db):
        # 如果没在段内，只有超过“开门阈值”才进入
        if not in_seg:
            if db >= thr_on_db:
                in_seg = True
                voiced[i] = True
        else:
            # 如果在段内，只要没低于“关门阈值”，就保持为人声
            if db >= thr_off_db:
                voiced[i] = True
            else:
                in_seg = False
                voiced[i] = False

    return voiced


# -----------------------------
# mask -> 片段索引列表
# -----------------------------
def mask_to_segments_idx(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs = []
    in_seg = False
    start = 0

    for i, v in enumerate(mask):
        # 从静音进入人声
        if v and not in_seg:
            in_seg = True
            start = i
        # 从人声进入静音
        elif (not v) and in_seg:
            in_seg = False
            segs.append((start, i))

    # 收尾：如果最后还在段内，补上结束
    if in_seg:
        segs.append((start, len(mask)))

    return segs


# -----------------------------
# 片段后处理：过滤短段、填平短静音、合并近段、扩边
# -----------------------------
def post_process_segments(
    segs_idx: List[Tuple[int, int]],
    hop_ms: float,
    min_voice_ms: float,
    fill_sil_ms: float,
    merge_gap_ms: float,
    pad_ms: float,
    num_frames: int,
) -> List[Tuple[int, int]]:
    # 把 ms 转换为帧数（向上取整更保守）
    min_voice_frames = int(np.ceil(min_voice_ms / hop_ms))
    fill_sil_frames = int(np.ceil(fill_sil_ms / hop_ms))
    merge_gap_frames = int(np.ceil(merge_gap_ms / hop_ms))
    pad_frames = int(np.ceil(pad_ms / hop_ms))

        # ---------- 0) 桥接合并：用“短段”把左右两段粘起来 ----------
    # 把 ms 转成帧数
    bridge_voice_frames = int(np.ceil(800.0 / hop_ms))   # 认为 <=0.8s 的段可能是“连接件”
    bridge_gap_frames   = int(np.ceil(250.0 / hop_ms))   # 左右 gap <=0.25s 就认为属于同一句

    # 如果段太少就不用桥接
    if len(segs_idx) >= 3:
        bridged = []
        i = 0
        while i < len(segs_idx):
            # 只要没到中间段就照常放
            if i == 0 or i == len(segs_idx) - 1:
                bridged.append(segs_idx[i])
                i += 1
                continue

            # 取前一段、当前段、后一段
            ps, pe = segs_idx[i - 1]
            cs, ce = segs_idx[i]
            ns, ne = segs_idx[i + 1]

            # 当前段长度（帧）
            cur_len = ce - cs

            # 当前段前后的 gap（帧）
            gap_left = cs - pe
            gap_right = ns - ce

            # 如果当前段很短且左右 gap 都很小，则把三段合并
            if cur_len <= bridge_voice_frames and gap_left <= bridge_gap_frames and gap_right <= bridge_gap_frames:
                # 用一个大段替换“前段 + 当前段 + 后段”
                merged_seg = (ps, ne)
                # 覆盖 bridged 最后一个（就是前段），然后跳过后段
                bridged[-1] = merged_seg
                i += 2
            else:
                # 否则保留当前段
                bridged.append(segs_idx[i])
                i += 1

        segs_idx = bridged


    # 1) 过滤太短的人声段（防止口水声/爆破音残片）
    segs_idx = [(s, e) for (s, e) in segs_idx if (e - s) >= min_voice_frames]
    if not segs_idx:
        return []

    # 2) 填平短静音：gap <= fill_sil_frames 就当作一句话里的停顿
    merged = [segs_idx[0]]
    for s, e in segs_idx[1:]:
        ps, pe = merged[-1]
        gap = s - pe
        if gap <= fill_sil_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # 3) 合并近邻段：gap <= merge_gap_frames 直接合并（进一步减少碎片）
    merged2 = [merged[0]]
    for s, e in merged[1:]:
        ps, pe = merged2[-1]
        gap = s - pe
        if gap <= merge_gap_frames:
            merged2[-1] = (ps, e)
        else:
            merged2.append((s, e))

    # 4) 扩边：把每段前后各加 pad_frames（保留起音、尾音、气声）
    padded = []
    for s, e in merged2:
        s2 = max(0, s - pad_frames)
        e2 = min(num_frames, e + pad_frames)
        padded.append((s2, e2))

    # 5) 扩边后可能产生重叠，再合并一次
    padded.sort()
    final = [padded[0]]
    for s, e in padded[1:]:
        ps, pe = final[-1]
        if s <= pe:
            final[-1] = (ps, max(pe, e))
        else:
            final.append((s, e))

    return final


# -----------------------------
# idx -> 秒
# -----------------------------
def idx_to_segments_time(segs_idx: List[Tuple[int, int]], times_s: np.ndarray, hop_ms: float) -> List[Segment]:
    hop_s = hop_ms / 1000.0
    segs = []

    for s, e in segs_idx:
        # 起点就是该帧时间
        start_s = float(times_s[s])
        # 终点用“最后一帧时间 + hop”补足，不然会少算一点
        end_i = min(e, len(times_s) - 1)
        end_s = float(times_s[end_i]) + hop_s
        segs.append(Segment(start_s, end_s))

    return segs


def main():
    ap = argparse.ArgumentParser(description="Analyze effective vocal duration in WAV for RVC (v2).")
    ap.add_argument("wav", help="Path to wav file")

    # 分帧参数：40/10ms 对人声很通用
    ap.add_argument("--frame_ms", type=float, default=40.0, help="Frame length in ms (default 40)")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="Hop length in ms (default 10)")

    # 阈值模式：auto/ manual
    ap.add_argument("--thr_mode", choices=["auto", "manual"], default="auto",
                    help="Threshold mode: auto or manual")
    ap.add_argument("--thr_db", type=float, default=-35.0,
                    help="Manual threshold in dBFS (only used if thr_mode=manual)")

    # auto 阈值参数：用高分位参考 + 相对下降量
    ap.add_argument("--ref_percentile", type=float, default=90.0,
                    help="Auto: reference percentile for vocal energy (default 90)")
    ap.add_argument("--relative_drop_db", type=float, default=35.0,
                    help="Auto: threshold = P(ref) - drop_db (default 35). Smaller keeps more breath/weak parts.")
    ap.add_argument("--clamp_min_db", type=float, default=-70.0,
                    help="Auto: minimum threshold clamp (default -70)")
    ap.add_argument("--clamp_max_db", type=float, default=-20.0,
                    help="Auto: maximum threshold clamp (default -20)")

    # 滞回：开门阈值/关门阈值差值（越大越不碎，但更可能把很轻的尾音吞掉）
    ap.add_argument("--hyst_db", type=float, default=3.0,
                    help="Hysteresis gap: thr_off = thr_on - hyst_db (default 3dB)")

    # 片段后处理参数：全是“减少碎片”相关
    ap.add_argument("--min_voice_ms", type=float, default=250.0,
                    help="Minimum voiced segment length (default 250ms)")
    ap.add_argument("--fill_sil_ms", type=float, default=180.0,
                    help="Fill silences shorter than this (default 180ms)")
    ap.add_argument("--merge_gap_ms", type=float, default=350.0,
                    help="Merge segments whose gap <= this (default 350ms)")
    ap.add_argument("--pad_ms", type=float, default=60.0,
                    help="Pad each segment on both sides (default 60ms)")

    # 输出：json
    ap.add_argument("--dump_json", type=str, default="",
                    help="If set, dump segments to json file")

    args = ap.parse_args()

    # 读 wav（支持 float / int），always_2d=False 方便处理
    x, sr = sf.read(args.wav, always_2d=False)

    # 转 mono，避免双声道能量不一致
    x = to_mono(x).astype(np.float32)

    # 如果数据不是 [-1,1] 的 float，做一次“只缩不放”的归一化保护
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > 1.0:
        x = x / peak

    # 计算帧级 RMS(dBFS)
    times_s, rms_db = frame_rms_db(x, sr, args.frame_ms, args.hop_ms)

    # 估计阈值（auto 会返回调试信息）
    thr_on_db, dbg = estimate_threshold_db(
        rms_db=rms_db,
        mode=args.thr_mode,
        manual_thr_db=args.thr_db,
        ref_percentile=args.ref_percentile,
        relative_drop_db=args.relative_drop_db,
        clamp_min_db=args.clamp_min_db,
        clamp_max_db=args.clamp_max_db,
    )

    # 关门阈值比开门阈值低一点，形成滞回
    thr_off_db = thr_on_db - float(args.hyst_db)

    # 做滞回 VAD（减少阈值附近抖动导致碎片）
    voiced_mask = vad_hysteresis(rms_db, thr_on_db=thr_on_db, thr_off_db=thr_off_db)

    # 把 mask 转成粗片段（帧索引）
    segs_idx = mask_to_segments_idx(voiced_mask)

    # 后处理：过滤短段、填平短静音、合并近邻、扩边
    segs_idx = post_process_segments(
        segs_idx=segs_idx,
        hop_ms=args.hop_ms,
        min_voice_ms=args.min_voice_ms,
        fill_sil_ms=args.fill_sil_ms,
        merge_gap_ms=args.merge_gap_ms,
        pad_ms=args.pad_ms,
        num_frames=len(times_s),
    )

    # 转为秒级片段
    segs = idx_to_segments_time(segs_idx, times_s, args.hop_ms)

    # 统计总时长
    total_effective = float(sum(s.dur_s for s in segs))
    total_duration = float(len(x) / sr)

    # 打印摘要
    print("==== RVC 有效人声时长分析 (v2) ====")
    print(f"文件: {args.wav}")
    print(f"采样率: {sr} Hz")
    print(f"总时长: {total_duration:.3f} s")

    if args.thr_mode == "auto":
        print("Auto 阈值信息:")
        print(f"  参考分位: P{args.ref_percentile:.1f} = {dbg['ref_db']:.2f} dBFS")
        print(f"  相对下降: -{dbg['relative_drop_db']:.1f} dB  -> thr_on = {thr_on_db:.2f} dBFS")
        print(f"  clamp: [{dbg['clamp_min_db']:.1f}, {dbg['clamp_max_db']:.1f}] dBFS")
    else:
        print(f"手动阈值: thr_on = {thr_on_db:.2f} dBFS")

    print(f"滞回: thr_off = {thr_off_db:.2f} dBFS  (hyst={args.hyst_db:.1f} dB)")
    print(f"有效片段数: {len(segs)}")
    print(f"有效人声总时长: {total_effective:.3f} s   ({(total_effective / total_duration * 100.0 if total_duration > 0 else 0):.1f}%)")

    # 片段分布统计（看碎片化情况）
    if segs:
        durs = np.array([s.dur_s for s in segs], dtype=np.float64)
        short_cnt = int(np.sum(durs < 1.0))
        print("片段时长统计:")
        print(f"  最短: {durs.min():.3f}s  最长: {durs.max():.3f}s  中位数: {np.median(durs):.3f}s")
        print(f"  <1.0s 的短片段数量: {short_cnt} / {len(segs)}")

        # 只展示前 30 段，避免刷屏
        print("片段列表(最多显示30段):")
        for i, s in enumerate(segs[:30]):
            print(f"  #{i+1:02d}: {s.start_s:8.3f}  -> {s.end_s:8.3f}   ({s.dur_s:6.3f}s)")
        if len(segs) > 30:
            print(f"  ... 还有 {len(segs) - 30} 段未显示")
        print("==== 分析结束 ====")
        print("BY OTREE")

    # 导出 json（方便拿去切片/训练流水线）
    if args.dump_json:
        out = {
            "wav": args.wav,
            "sr": sr,
            "duration_s": total_duration,
            "thr_mode": args.thr_mode,
            "thr_on_db": thr_on_db,
            "thr_off_db": thr_off_db,
            "params": {
                "frame_ms": args.frame_ms,
                "hop_ms": args.hop_ms,
                "ref_percentile": args.ref_percentile,
                "relative_drop_db": args.relative_drop_db,
                "clamp_min_db": args.clamp_min_db,
                "clamp_max_db": args.clamp_max_db,
                "hyst_db": args.hyst_db,
                "min_voice_ms": args.min_voice_ms,
                "fill_sil_ms": args.fill_sil_ms,
                "merge_gap_ms": args.merge_gap_ms,
                "pad_ms": args.pad_ms,
            },
            "segments": [{"start_s": s.start_s, "end_s": s.end_s, "dur_s": s.dur_s} for s in segs],
            "effective_total_s": total_effective,
        }
        if dbg is not None:
            out["auto_debug"] = dbg

        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"已导出 JSON: {args.dump_json}")


if __name__ == "__main__":
    main()
