    #!/usr/bin/env python3
    # Conv trainer for this project

    import os, glob, json, time, datetime, random
    import numpy as np
    import tensorflow as tf
    from keras import layers as KL, models as KM, optimizers as KO
    from keras.saving import register_keras_serializable

    # config
    DATA_GLOB = os.getenv("DATA_GLOB", "shards/1200_bin-*.tfrecord.gz")
    RUN_TAG   = os.getenv("RUN_TAG", "bin1200_main")
    EPOCHS    = int(os.getenv("EPOCHS", "3"))

    # Fractions of total shards to val and test set
    VAL_FRAC  = float(os.getenv("VAL_FRAC", "0.02"))
    TEST_FRAC = float(os.getenv("TEST_FRAC", "0.02"))

    BATCH     = int(os.getenv("BATCH", "0"))      
    BASE_LR   = float(os.getenv("BASE_LR", "3e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    SHARD_SIZE_EST = int(os.getenv("SHARD_SIZE_EST", "100000"))

    MIXED_PRECISION = os.getenv("MIXED_PRECISION", "1").lower() not in ("0","false")
    USE_XLA   = os.getenv("USE_XLA", "1").lower() not in ("0","false")
    STEPS_PER_EXECUTION = int(os.getenv("STEPS_PER_EXECUTION", "2048"))

    RUNS_DIR  = os.getenv("RUNS_DIR", "runs_main")
    MODELS_DIR= os.getenv("MODELS_DIR", "models_main")
    SEED      = int(os.getenv("SEED", "42"))
    RESUME_FROM = os.getenv("RESUME_FROM", "") 

   
    INPUT_SHAPE = (8, 8, 36)
    LOGITS      = 64 * 64 * 5
    MASK_BYTES  = (LOGITS + 7) // 8
    _STARTS_10  = tf.constant([8.,2.,2.,2.,1.,  8.,2.,2.,2.,1.], tf.float32)

    # TFRecord parsing
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

        # Repair channels 12–21 ("any captured" flags) on the fly, as explained, newproc's none captured was revised as such
        present10 = tf.concat([x[:, :, 0:5], x[:, :, 6:11]], axis=-1)
        any_missing10 = tf.cast(tf.reduce_sum(present10, axis=[0,1]) < _STARTS_10, tf.float32)
        x = tf.concat([x[:, :, :12], tf.tile(any_missing10[None, None, :], [8, 8, 1]), x[:, :, 22:]], axis=-1)

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

    # model 
    def conv_block(x, ch):
        x = KL.Conv2D(ch, 3, padding="same", use_bias=False)(x)
        x = KL.BatchNormalization()(x)
        return KL.ReLU()(x)

    def build_conv_body():
        i = KL.Input(shape=INPUT_SHAPE, name="board")
        x = conv_block(i, 64); x = conv_block(x, 128); x = conv_block(x, 192)
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

    # LR schedule
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

    # helpers
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
        return f"{RUN_TAG}_conv_ep{EPOCHS}_bs{BATCH}_lr{BASE_LR:g}_wd{WEIGHT_DECAY:g}_{ts}"

    # main 
    def main():
        os.makedirs(RUNS_DIR, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
        os.environ.setdefault("TF_GPU_ALLOCATOR","cuda_malloc_async")
        if USE_XLA: tf.config.optimizer.set_jit(True)
        if MIXED_PRECISION: tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

        files = sorted(glob.glob(DATA_GLOB))
        if len(files) < 4:
            raise SystemExit(f"Need at least 4 shards for train/val/test. Found {len(files)} for {DATA_GLOB}")

        # Seeded shuffle + split
        random.shuffle(files)
        n_test = max(1, int(len(files) * TEST_FRAC))
        test_files = files[:n_test]
        remainder = files[n_test:]
        n_val = max(1, int(len(files) * VAL_FRAC)) 
        val_files = remainder[:n_val]
        train_files = remainder[n_val:]

        run_id  = _run_id()
        logdir  = os.path.join(RUNS_DIR, run_id)
        outbase = os.path.join(MODELS_DIR, run_id)

        batch = pick_batch(train_files)

        print("\n" + "="*80)
        print(f"[info] Run: {run_id}")
        print(f"[cfg]  Epochs={EPOCHS}  Batch={batch}  BaseLR={BASE_LR}  WD={WEIGHT_DECAY}")
        print(f"[cfg]  XLA={USE_XLA}  AMP={MIXED_PRECISION}  steps/exe={STEPS_PER_EXECUTION}  Seed={SEED}")
        print(f"[data] Shards: train={len(train_files)}  val={len(val_files)}  test={len(test_files)}")
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

        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "config.json"), "w") as fh:
            json.dump({
                "run_id": run_id, "data_glob": DATA_GLOB,
                "val_frac_total": VAL_FRAC, "test_frac_total": TEST_FRAC,
                "epochs": EPOCHS, "batch": batch, "base_lr": BASE_LR, "weight_decay": WEIGHT_DECAY,
                "mixed_precision": MIXED_PRECISION, "xla": USE_XLA,
                "steps_per_execution": STEPS_PER_EXECUTION,
                "steps_per_epoch_est": int(steps_per_epoch_est), "total_steps_est": int(total_steps),
                "seed": SEED
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

        # final test eval on best checkpoint, original run did not log these metrics, test.py is provided along with this script to show this
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
        print("[test] Final test metrics (loss, top1, top3, top5):", res)

    if __name__ == "__main__":
        main()