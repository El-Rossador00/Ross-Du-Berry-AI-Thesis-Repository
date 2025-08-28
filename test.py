#I did not log test metrics in train_main, however the test set is reproduceable, as implemented here

#!/usr/bin/env python3
import os, glob, json, random, datetime
import numpy as np
import tensorflow as tf
from keras import layers as KL, models as KM, optimizers as KO
from keras.saving import register_keras_serializable
import pandas as pd

DATA_GLOB = os.getenv("DATA_GLOB", "shards/1200_bin-*.tfrecord.gz")
RUN_TAG = os.getenv("RUN_TAG", "conv1200")
BATCH = int(os.getenv("BATCH", "0"))
RUNS_DIR = os.getenv("RUNS_DIR", "runs_main")
MODELS_DIR = os.getenv("MODELS_DIR", "models_main")
SEED = int(os.getenv("SEED", "42"))
TEST_FRAC = float(os.getenv("TEST_FRAC", "0.02"))
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "")

INPUT_SHAPE = (8, 8, 36)
LOGITS = 64 * 64 * 5
MASK_BYTES = (LOGITS + 7) // 8
_STARTS_10 = tf.constant([8.,2.,2.,2.,1., 8.,2.,2.,2.,1.], tf.float32)

_feature_spec = {
    "tensor": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "legal": tf.io.FixedLenFeature([], tf.string),
}

def _unpack_bits_le(ubytes):
    u = tf.cast(ubytes, tf.uint8)
    u = tf.expand_dims(u, -1)
    shifts = tf.reshape(tf.constant([0,1,2,3,4,5,6,7], dtype=tf.uint8), [1,8])
    shifted = tf.bitwise.right_shift(u, shifts)
    bits = tf.bitwise.bitwise_and(shifted, tf.constant(1, tf.uint8))
    return bits

def _parse_example(ex):
    f = tf.io.parse_single_example(ex, _feature_spec)
    x = tf.io.decode_raw(f["tensor"], tf.float32)
    x = tf.reshape(x, INPUT_SHAPE)
    x = tf.ensure_shape(x, INPUT_SHAPE)
    present10 = tf.concat([x[:, :, 0:5], x[:, :, 6:11]], axis=-1)
    any_missing10 = tf.cast(tf.reduce_sum(present10, axis=[0,1]) < _STARTS_10, tf.float32)
    x = tf.concat([x[:, :, :12], tf.tile(any_missing10[None, None, :], [8, 8, 1]), x[:, :, 22:]], axis=-1)
    legal_bytes = tf.io.decode_raw(f["legal"], tf.uint8)
    legal_bytes = tf.ensure_shape(legal_bytes, [MASK_BYTES])
    bits = _unpack_bits_le(legal_bytes)
    legal = tf.cast(tf.reshape(bits, [-1])[:LOGITS], tf.float32)
    y = tf.cast(f["label"], tf.int32)
    return ({"board": x, "mask": legal}, y)

def make_dataset(files, batch, shuffle_files=False, shuffle_buf=0):
    if shuffle_files:
        files = tf.random.shuffle(files, seed=SEED)
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False)
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    return ds.with_options(opts).prefetch(tf.data.AUTOTUNE)


def conv_block(x, ch):
    x = KL.Conv2D(ch, 3, padding="same", use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    return KL.ReLU()(x)

def build_conv_body():
    i = KL.Input(shape=INPUT_SHAPE, name="board")
    x = conv_block(i, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 192)
    x = KL.Flatten()(x)
    x = KL.Dense(2048, activation="relu")(x)
    logits = KL.Dense(LOGITS, dtype="float32")(x)
    return i, logits

@register_keras_serializable(package="masking")
def _apply_mask(inputs):
    mask, z = inputs
    minus_inf = tf.cast(-1e9, z.dtype)
    return tf.where(mask > 0.5, z, tf.fill(tf.shape(z), minus_inf))

def build_model():
    mask_in = KL.Input(shape=(LOGITS,), name="mask", dtype="float32")
    board_in, logits = build_conv_body()
    masked = KL.Lambda(_apply_mask, name="apply_mask")([mask_in, logits])
    out = KL.Activation("linear", dtype="float32")(masked)
    return KM.Model(inputs={"board": board_in, "mask": mask_in}, outputs=out)


def pick_batch(test_files):
    if BATCH and BATCH > 0:
        print(f"[auto-batch] Using user batch={BATCH}")
        return BATCH
    print("[auto-batch] Probing batch…")
    for b in [4096, 3072, 2048]:
        try:
            ds = make_dataset(test_files[:max(1, len(test_files)//4)], batch=b)
            _ = next(iter(ds.take(1)))
            print(f"[auto-batch] OK at batch={b}")
            return b
        except Exception as e:
            print(f"[auto-batch] batch={b} failed: {e}")
    return 1024

def _run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{RUN_TAG}_test_only_{ts}"

def main():
    if not BEST_MODEL_PATH or not os.path.exists(BEST_MODEL_PATH):
        raise SystemExit(f"Best model path '{BEST_MODEL_PATH}' not found. Set BEST_MODEL_PATH environment variable.")

    os.makedirs(RUNS_DIR, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    files = sorted(glob.glob(DATA_GLOB))
    if len(files) < 4:
        raise SystemExit(f"Need at least 4 shards for test. Found {len(files)} for {DATA_GLOB}")

    random.shuffle(files)
    n_test = max(1, int(len(files) * TEST_FRAC))
    test_files = files[:n_test]

    run_id = _run_id()
    logdir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(logdir, exist_ok=True)

    batch = pick_batch(test_files)
    test_ds = make_dataset(test_files, batch=batch, shuffle_files=False, shuffle_buf=0)

    print("\n" + "="*80)
    print(f"[info] Run: {run_id}")
    print(f"[cfg]  Batch={batch}  Seed={SEED}")
    print(f"[data] Test shards={len(test_files)}")
    print(f"[model] Loading from {BEST_MODEL_PATH}")
    print("="*80 + "\n")

    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH,
        compile=False,
        custom_objects={'_apply_mask': _apply_mask}
    )
    best_model.compile(
        optimizer=KO.AdamW(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ]
    )

    print("[test] Evaluating best model on test set…")
    res = best_model.evaluate(test_ds, verbose=1)
    print("[test] Final test metrics (loss, top1, top3, top5):", res)

    # Save test metrics
    with open(os.path.join(logdir, "test_metrics.json"), "w") as fh:
        json.dump({
            "test_loss": float(res[0]),
            "test_top1": float(res[1]),
            "test_top3": float(res[2]),
            "test_top5": float(res[3]),
        }, fh, indent=2)

    test_df = pd.DataFrame({
        "epoch": ["test"],
        "loss": [res[0]],
        "top1": [res[1]],
        "top3": [res[2]],
        "top5": [res[3]],
    })
    test_df.to_csv(os.path.join(logdir, "test_metrics.csv"), index=False)

    with tf.summary.create_file_writer(logdir).as_default():
        tf.summary.scalar("test/loss", res[0], step=0)
        tf.summary.scalar("test/top1", res[1], step=0)
        tf.summary.scalar("test/top3", res[2], step=0)
        tf.summary.scalar("test/top5", res[3], step=0)

if __name__ == "__main__":
    main()
