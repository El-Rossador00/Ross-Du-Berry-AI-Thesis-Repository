#!/usr/bin/env python3
# Evaluate Keras per-bin models on eval_sets/{bin}_eval-00000-of-00001.tfrecord.gz
import os, json, csv, glob
from pathlib import Path
from typing import List, Dict
import numpy as np
import tensorflow as tf
from keras import optimizers as KO
from keras.saving import register_keras_serializable

def _csv_ints(s, default):
    raw = os.getenv(s, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

BINS = _csv_ints("BINS", "1200,1500,1800,2000")
EVAL_DIR = os.getenv("EVAL_DIR", "eval_sets")
OUT_DIR = os.getenv("OUT_DIR", "model_eval_on_evalsets")
MODEL_TEMPLATE = os.getenv("MODEL_TEMPLATE", "modmod/conv{bin}_*_best.keras")
MODEL_PATH = os.getenv("MODEL_PATH", "") 
BATCH_PRED = int(os.getenv("BATCH_PRED", "1024"))

INPUT_SHAPE = (8, 8, 36)
LOGITS = 64 * 64 * 5
MASK_BYTES = (LOGITS + 7) // 8
_STARTS_10 = tf.constant([8.,2.,2.,2.,1.,  8.,2.,2.,2.,1.], tf.float32)

_FEATURE_SPEC = {
    "tensor": tf.io.FixedLenFeature([], tf.string),
    "label":  tf.io.FixedLenFeature([], tf.int64),
    "legal":  tf.io.FixedLenFeature([], tf.string),
}

def _unpack_bits_le(ubytes):
    u = tf.cast(ubytes, tf.uint8)
    u = tf.expand_dims(u, -1)
    shifts = tf.reshape(tf.constant([0,1,2,3,4,5,6,7], dtype=tf.uint8), [1,8])
    shifted = tf.bitwise.right_shift(u, shifts)
    bits = tf.bitwise.bitwise_and(shifted, tf.constant(1, tf.uint8))
    return bits

def _parse_example(ex):
    f = tf.io.parse_single_example(ex, _FEATURE_SPEC)

    x = tf.io.decode_raw(f["tensor"], tf.float32)
    x = tf.reshape(x, INPUT_SHAPE)
    x = tf.ensure_shape(x, INPUT_SHAPE)

    # repair channels 12–21 (parity with trainer)
    present10 = tf.concat([x[:, :, 0:5], x[:, :, 6:11]], axis=-1)
    any_missing10 = tf.cast(tf.reduce_sum(present10, axis=[0,1]) < _STARTS_10, tf.float32)
    x = tf.concat([x[:, :, :12], tf.tile(any_missing10[None, None, :], [8, 8, 1]), x[:, :, 22:]], axis=-1)

    legal_bytes = tf.io.decode_raw(f["legal"], tf.uint8)
    legal_bytes = tf.ensure_shape(legal_bytes, [MASK_BYTES])
    bits = _unpack_bits_le(legal_bytes)
    legal = tf.cast(tf.reshape(bits, [-1])[:LOGITS], tf.float32)

    y = tf.cast(f["label"], tf.int32)
    return ({"board": x, "mask": legal}, y)

@register_keras_serializable(package="masking")
def _apply_mask(inputs):
    mask, z = inputs
    minus_inf = tf.cast(-1e9, z.dtype)
    return tf.where(mask > 0.5, z, tf.fill(tf.shape(z), minus_inf))

def _resolve_model_path(elo_bin: int) -> str:
    if MODEL_TEMPLATE:
        cand = MODEL_TEMPLATE.format(bin=elo_bin)
        if "*" in cand:
            matches = sorted(glob.glob(cand))
            if not matches:
                raise FileNotFoundError(f"No model matches: {cand}")
            return matches[-1]
        if not os.path.exists(cand):
            raise FileNotFoundError(f"Model not found: {cand}")
        return cand
    if MODEL_PATH:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
        return MODEL_PATH
    raise RuntimeError("Provide MODEL_TEMPLATE or MODEL_PATH")

def _load_model(path: str):
    m = tf.keras.models.load_model(path, compile=False, custom_objects={'_apply_mask': _apply_mask})
    # compile just to stabilize metric containers (optimizer won't run training)
    m.compile(optimizer=KO.AdamW(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return m

def eval_one_bin(elo_bin: int) -> Dict:
    tfr = Path(EVAL_DIR) / f"{elo_bin}_eval-00000-of-00001.tfrecord.gz"
    if not tfr.exists():
        print(f"[bin {elo_bin}] Missing eval TFRecord: {tfr}")
        return {}

    ds = (tf.data.TFRecordDataset(str(tfr), compression_type="GZIP")
          .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_PRED)
          .prefetch(tf.data.AUTOTUNE))

    model_path = _resolve_model_path(elo_bin)
    print(f"[bin {elo_bin}] Model: {model_path}")
    model = _load_model(model_path)

    matched = 0
    top1 = top3 = top5 = 0

    for (features, y) in ds:
        logits = model.predict({"board": features["board"], "mask": features["mask"]}, verbose=0)
        y_np = y.numpy()

        # top-1
        top1_idx = np.argmax(logits, axis=1)
        top1 += int(np.sum(top1_idx == y_np))

        # top-3 / top-5
        top3_idx = np.argpartition(logits, -3, axis=1)[:, -3:]
        top5_idx = np.argpartition(logits, -5, axis=1)[:, -5:]
        for i in range(len(y_np)):
            lab = y_np[i]
            if lab in top3_idx[i]: top3 += 1
            if lab in top5_idx[i]: top5 += 1

        matched += len(y_np)

    if matched == 0:
        print(f"[bin {elo_bin}] No examples read.")
        return {}

    res = {
        "bin": elo_bin,
        "positions": matched,
        "top1_accuracy": top1 / matched,
        "top3_accuracy": top3 / matched,
        "top5_accuracy": top5 / matched,
        "model_path": model_path,
    }
    print(f"[bin {elo_bin}] positions={matched:,}  "
          f"top1={res['top1_accuracy']:.4f}  top3={res['top3_accuracy']:.4f}  top5={res['top5_accuracy']:.4f}")
    return res

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_res = []
    for b in BINS:
        r = eval_one_bin(b)
        if r: all_res.append(r)
    if all_res:
        for r in all_res:
            with open(Path(OUT_DIR) / f"model_eval_bin{r['bin']}.json", "w") as fh:
                json.dump(r, fh, indent=2)
        csv_path = Path(OUT_DIR) / "summary_models.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_res[0].keys()))
            writer.writeheader(); writer.writerows(all_res)
        print(f"[done] Combined summary -> {csv_path}")

if __name__ == "__main__":
    main()
