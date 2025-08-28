#Modifies train_main for ablations testing
#!/usr/bin/env python3
import os, glob, json, time, datetime, random, csv, math
import numpy as np
import tensorflow as tf
from keras import layers as KL, models as KM, optimizers as KO
from keras.saving import register_keras_serializable

DATA_GLOB   = os.getenv("DATA_GLOB", "shards/1500_bin-*.tfrecord.gz")
RUN_TAG     = os.getenv("RUN_TAG", "bin1500")
RUN_LABEL   = os.getenv("RUN_LABEL", "ablate")
EPOCHS      = int(os.getenv("EPOCHS", "8"))
VAL_FRAC    = float(os.getenv("VAL_FRAC", "0.02"))
TEST_FRAC   = float(os.getenv("TEST_FRAC", "0.02"))
BATCH       = int(os.getenv("BATCH", "4096"))
BASE_LR     = float(os.getenv("BASE_LR", "3e-4"))
WEIGHT_DECAY= float(os.getenv("WEIGHT_DECAY", "1e-4"))
SHARD_SIZE_EST = int(os.getenv("SHARD_SIZE_EST", "25000"))
TRAIN_SHARD_CAP = int(os.getenv("TRAIN_SHARD_CAP", "8"))
VAL_SHARD_CAP   = int(os.getenv("VAL_SHARD_CAP", "2"))
TEST_SHARD_CAP  = int(os.getenv("TEST_SHARD_CAP", "2"))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "1").lower() not in ("0","false")
USE_XLA   = os.getenv("USE_XLA", "1").lower() not in ("0","false")
STEPS_PER_EXECUTION = int(os.getenv("STEPS_PER_EXECUTION", "2048"))
RUNS_DIR  = os.getenv("RUNS_DIR", "runs_ablate")
MODELS_DIR= os.getenv("MODELS_DIR", "models_ablate")
SEED      = int(os.getenv("SEED", "42"))
RESUME_FROM = os.getenv("RESUME_FROM", "")
MODEL_BODY= os.getenv("MODEL_BODY", "conv")
ABLATE_ANY_CAPTURED_REPAIR = os.getenv("ABLATE_ANY_CAPTURED_REPAIR", "1") not in ("0","false")
ABLATE_DROP_MISSING   = os.getenv("ABLATE_DROP_MISSING", "0") not in ("0","false")
ABLATE_DROP_MATBAL    = os.getenv("ABLATE_DROP_MATBAL", "0") not in ("0","false")
ABLATE_DROP_TEMPORAL  = os.getenv("ABLATE_DROP_TEMPORAL", "0") not in ("0","false")
ABLATE_DROP_TACTICAL  = os.getenv("ABLATE_DROP_TACTICAL", "0") not in ("0","false")
ABLATE_DROP_CASTLING  = os.getenv("ABLATE_DROP_CASTLING", "0") not in ("0","false")
ABLATE_NO_MASKING     = os.getenv("ABLATE_NO_MASKING", "0") not in ("0","false")
INPUT_SHAPE = (8, 8, 36)
LOGITS      = 64 * 64 * 5
MASK_BYTES  = (LOGITS + 7) // 8
_STARTS_10  = tf.constant([8.,2.,2.,2.,1.,  8.,2.,2.,2.,1.], tf.float32)
_feature_spec = {
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
    f = tf.io.parse_single_example(ex, _feature_spec)
    x = tf.io.decode_raw(f["tensor"], tf.float32)
    x = tf.reshape(x, INPUT_SHAPE)
    x = tf.ensure_shape(x, INPUT_SHAPE)
    if ABLATE_ANY_CAPTURED_REPAIR:
        present10 = tf.concat([x[:, :, 0:5], x[:, :, 6:11]], axis=-1)
        any_missing10 = tf.cast(tf.reduce_sum(present10, axis=[0,1]) < _STARTS_10, tf.float32)
        repaired = tf.tile(any_missing10[None, None, :], [8, 8, 1])
        x = tf.concat([x[:, :, :12], repaired, x[:, :, 22:]], axis=-1)
    def zero_planes(t, lo, hi_inclusive):
        pads = []
        if lo > 0:
            pads.append(t[:, :, :lo])
        pads.append(tf.zeros_like(t[:, :, lo:hi_inclusive+1]))
        if hi_inclusive + 1 < t.shape[-1]:
            pads.append(t[:, :, hi_inclusive+1:])
        return tf.concat(pads, axis=-1)
    if ABLATE_DROP_MISSING:   x = zero_planes(x, 12, 21)
    if ABLATE_DROP_MATBAL:    x = zero_planes(x, 35, 35)
    if ABLATE_DROP_TEMPORAL:
        for ch in [28,29,30,31,34]:
            x = zero_planes(x, ch, ch)
    if ABLATE_DROP_TACTICAL:
        for ch in [27,32,33]:
            x = zero_planes(x, ch, ch)
    if ABLATE_DROP_CASTLING:
        x = zero_planes(x, 23, 26)
    legal_bytes = tf.io.decode_raw(f["legal"], tf.uint8)
    legal_bytes = tf.ensure_shape(legal_bytes, [MASK_BYTES])
    bits = _unpack_bits_le(legal_bytes)
    legal = tf.cast(tf.reshape(bits, [-1])[:LOGITS], tf.float32)
    y = tf.cast(f["label"], tf.int32)
    return ({"board": x, "mask": legal}, y)
def make_dataset(files, batch, shuffle_files=True, shuffle_buf=200_000):
    if shuffle_files:
        files = tf.random.shuffle(files, seed=SEED)
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=False)
    opts = tf.data.Options(); opts.experimental_deterministic = False
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
def build_vit_body(d_model=256, num_layers=4, num_heads=8, mlp_ratio=4):
    i = KL.Input(shape=INPUT_SHAPE, name="board")
    x = KL.Conv2D(d_model, kernel_size=1, use_bias=False)(i)
    x = KL.Reshape((64, d_model))(x)
    pos = tf.range(64)[None, :]
    pos_emb = KL.Embedding(input_dim=64, output_dim=d_model)(pos)
    x = x + pos_emb
    for _ in range(num_layers):
        x1 = KL.LayerNormalization(epsilon=1e-6)(x)
        x1 = KL.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=0.0)(x1, x1)
        x = KL.Add()([x, x1])
        x2 = KL.LayerNormalization(epsilon=1e-6)(x)
        x2 = KL.Dense(d_model*mlp_ratio, activation="gelu")(x2)
        x2 = KL.Dense(d_model)(x2)
        x = KL.Add()([x, x2])
    x = KL.LayerNormalization(epsilon=1e-6)(x)
    x = KL.Flatten()(x)
    logits = KL.Dense(LOGITS, dtype="float32")(x)
    return i, logits
@register_keras_serializable(package="masking")
def _apply_mask(inputs):
    mask, z = inputs
    minus_inf = tf.cast(-1e9, z.dtype)
    return tf.where(mask > 0.5, z, tf.fill(tf.shape(z), minus_inf))
def build_model():
    mask_in = KL.Input(shape=(LOGITS,), name="mask", dtype="float32")
    if MODEL_BODY.lower() == "vit":
        board_in, logits = build_vit_body()
    else:
        board_in, logits = build_conv_body()
    if ABLATE_NO_MASKING:
        out = KL.Activation("linear", dtype="float32")(logits)
    else:
        masked = KL.Lambda(_apply_mask, name="apply_mask")([mask_in, logits])
        out = KL.Activation("linear", dtype="float32")(masked)
    return KM.Model(inputs={"board": board_in, "mask": mask_in}, outputs=out)
class StepLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total = int(max(1, total_steps))
        self.warm = int(max(1, warmup_steps))
    def _lr_at(self, step):
        s = float(step)
        if s < self.warm: return self.base_lr * (s + 1.0) / self.warm
        t = (s - self.warm) / max(1.0, (self.total - self.warm))
        return 0.5 * self.base_lr * (1.0 + np.cos(np.pi * t))
    def on_train_batch_begin(self, batch, logs=None):
        step = int(tf.keras.backend.get_value(self.model.optimizer.iterations))
        self.model.optimizer.learning_rate.assign(self._lr_at(step))
class LrToLogs(tf.keras.callbacks.Callback):
    def __init__(self, logdir):
        super().__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(logdir, "lr"))
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if logs is not None: logs["lr"] = lr
        with self.writer.as_default():
            tf.summary.scalar("learning_rate", lr, step=epoch)
def pick_batch(train_files):
    if BATCH and BATCH > 0:
        print(f"[auto-batch] Using user batch={BATCH}")
        return BATCH
    print("[auto-batch] Probing batch…")
    for b in [4096, 3072, 2048]:
        try:
            ds = make_dataset(train_files[:max(1, len(train_files)//4)], batch=b)
            _ = next(iter(ds.take(1)))
            print(f"[auto-batch] OK at batch={b}")
            return b
        except Exception as e:
            print(f"[auto-batch] batch={b} failed: {e}")
    return 1024
def _run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    body = "vit" if MODEL_BODY.lower() == "vit" else "conv"
    return f"{RUN_LABEL}_{body}_ep{EPOCHS}_bs{BATCH}_lr{BASE_LR:g}_wd{WEIGHT_DECAY:g}_{ts}"
def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
def _append_test_row_to_history(history_csv, test_dict):
    if not os.path.exists(history_csv):
        return
    with open(history_csv, "r") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    test_row = {h: "" for h in header}
    test_row["epoch"] = "test"
    if "loss" in header and "loss" in test_dict:          test_row["loss"] = ""
    if "top1" in header and "top1" in test_dict:          test_row["top1"] = ""
    if "top3" in header and "top3" in test_dict:          test_row["top3"] = ""
    if "top5" in header and "top5" in test_dict:          test_row["top5"] = ""
    if "val_loss" in header:                              test_row["val_loss"] = ""
    if "val_top1" in header:                              test_row["val_top1"] = ""
    if "val_top3" in header:                              test_row["val_top3"] = ""
    if "val_top5" in header:                              test_row["val_top5"] = ""
    for k in ("loss","top1","top3","top5"):
        col = f"test_{k}"
        if col not in header:
            header.append(col)
        test_row[col] = f"{test_dict.get(k, '')}"
    with open(history_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows[1:]:
            writer.writerow(r)
        writer.writerow([test_row.get(h, "") for h in header])
def main():
    _ensure_dirs(RUNS_DIR, MODELS_DIR)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
    os.environ.setdefault("TF_GPU_ALLOCATOR","cuda_malloc_async")
    if USE_XLA: tf.config.optimizer.set_jit(True)
    if MIXED_PRECISION: tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    files = sorted(glob.glob(DATA_GLOB))
    if len(files) < 4:
        raise SystemExit(f"Need at least 4 shards for train/val/test. Found {len(files)} for {DATA_GLOB}")
    random.shuffle(files)
    n_test = max(1, int(len(files) * TEST_FRAC))
    test_files = files[:n_test]
    remainder = files[n_test:]
    n_val = max(1, int(len(files) * VAL_FRAC))
    val_files = remainder[:n_val]
    train_files = remainder[n_val:]
    if TRAIN_SHARD_CAP > 0: train_files = train_files[:min(len(train_files), TRAIN_SHARD_CAP)]
    if VAL_SHARD_CAP   > 0: val_files   = val_files[:min(len(val_files), VAL_SHARD_CAP)]
    if TEST_SHARD_CAP  > 0: test_files  = test_files[:min(len(test_files), TEST_SHARD_CAP)]
    run_id  = _run_id()
    logdir  = os.path.join(RUNS_DIR, run_id)
    outbase = os.path.join(MODELS_DIR, run_id)
    _ensure_dirs(logdir, os.path.dirname(outbase))
    batch = pick_batch(train_files)
    print("\n" + "="*80)
    print(f"[info] Run: {run_id}")
    print(f"[cfg]  Body={MODEL_BODY}  Epochs={EPOCHS}  Batch={batch}  BaseLR={BASE_LR}  WD={WEIGHT_DECAY}")
    print(f"[cfg]  XLA={USE_XLA}  AMP={MIXED_PRECISION}  steps/exe={STEPS_PER_EXECUTION}  Seed={SEED}")
    print(f"[abl]  any_captured={int(ABLATE_ANY_CAPTURED_REPAIR)}  no_masking={int(ABLATE_NO_MASKING)}")
    print(f"[abl]  missing={int(ABLATE_DROP_MISSING)}  matbal={int(ABLATE_DROP_MATBAL)}  temporal={int(ABLATE_DROP_TEMPORAL)}  tactical={int(ABLATE_DROP_TACTICAL)}  castling={int(ABLATE_DROP_CASTLING)}")
    print(f"[data] Shards (post-cap): train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")
    print("="*80 + "\n")
    train_ds = make_dataset(train_files, batch=batch, shuffle_files=True)
    val_ds   = make_dataset(val_files,   batch=batch, shuffle_files=False, shuffle_buf=0)
    test_ds  = make_dataset(test_files,  batch=batch, shuffle_files=False, shuffle_buf=0)
    steps_per_epoch_est = max(1, (len(train_files) * SHARD_SIZE_EST) // batch)
    total_steps = steps_per_epoch_est * EPOCHS
    warmup = max(1000, int(0.03 * total_steps))
    print(f"[schedule] Steps/epoch≈{steps_per_epoch_est:,}  Total≈{total_steps:,}  Warmup={warmup:,}")
    opt = KO.AdamW(learning_rate=BASE_LR, weight_decay=WEIGHT_DECAY, global_clipnorm=1.0)
    model = build_model()
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
        steps_per_execution=STEPS_PER_EXECUTION
    )
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"[init] Loading weights from {RESUME_FROM}")
        prev = tf.keras.models.load_model(
            RESUME_FROM, compile=False, custom_objects={'_apply_mask': _apply_mask}
        )
        model.set_weights(prev.get_weights())
    with open(os.path.join(logdir, "config.json"), "w") as fh:
        json.dump({
            "run_id": run_id, "data_glob": DATA_GLOB,
            "val_frac_total": VAL_FRAC, "test_frac_total": TEST_FRAC,
            "epochs": EPOCHS, "batch": batch, "base_lr": BASE_LR, "weight_decay": WEIGHT_DECAY,
            "mixed_precision": MIXED_PRECISION, "xla": USE_XLA,
            "steps_per_execution": STEPS_PER_EXECUTION,
            "steps_per_epoch_est": int(steps_per_epoch_est), "total_steps_est": int(total_steps),
            "seed": SEED,
            "caps": {"train": TRAIN_SHARD_CAP, "val": VAL_SHARD_CAP, "test": TEST_SHARD_CAP},
            "model_body": MODEL_BODY,
            "ablations": {
                "any_captured_repair": int(ABLATE_ANY_CAPTURED_REPAIR),
                "drop_missing": int(ABLATE_DROP_MISSING),
                "drop_matbal": int(ABLATE_DROP_MATBAL),
                "drop_temporal": int(ABLATE_DROP_TEMPORAL),
                "drop_tactical": int(ABLATE_DROP_TACTICAL),
                "drop_castling": int(ABLATE_DROP_CASTLING),
                "no_masking": int(ABLATE_NO_MASKING),
            }
        }, fh, indent=2)
    cb = [
        tf.keras.callbacks.ModelCheckpoint(outbase + "_epoch-{epoch:02d}.keras", save_best_only=False, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(outbase + "_best.keras", monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, histogram_freq=0),
        tf.keras.callbacks.CSVLogger(os.path.join(logdir, "history.csv")),
        StepLRSchedule(BASE_LR, total_steps=total_steps, warmup_steps=warmup),
        LrToLogs(logdir),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=1)
    print("\n[final] Loading best model for test eval…")
    best_model = tf.keras.models.load_model(
        outbase + "_best.keras",
        compile=False,
        custom_objects={'_apply_mask': _apply_mask}
    )
    best_model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ]
    )
    print("[test] Evaluating best on test set…")
    res = best_model.evaluate(test_ds, verbose=1)
    keys = ["loss", "top1", "top3", "top5"]
    test_dict = {k: float(v) for k, v in zip(keys, res)}
    print("[test] Final test metrics:", test_dict)
    with open(os.path.join(logdir, "test_metrics.json"), "w") as fh:
        json.dump(test_dict, fh, indent=2)
    with open(os.path.join(logdir, "test_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys); w.writerow([test_dict[k] for k in keys])
    _append_test_row_to_history(os.path.join(logdir, "history.csv"), test_dict)
    print("[test] Saved test_metrics.json and test_metrics.csv")
    print("[test] Appended test row into history.csv")
if __name__ == "__main__":
    main()
