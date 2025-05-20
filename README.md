For training a model with VGG19 conv2_2 and a random forest classifier.

```bash
pip install .
```

```bash
pitcount-cfim-training 
    -h
    --config-index CONFIG_INDEX
    --config-options CONFIG_OPTIONS
    --input-dir INPUT_DIR
    --output-dir OUTPUT_DIR
    --model-dir MODEL_DIR
```
mby ill expand this later


Bulk run command
```bash
python training_package\bulk_pipeline.py --n_estimators 10 20 50 64 --max_depth 10 20 --feature_limit 20 30 0.3 0.5 --n_jobs -1 --n_components 10 20 30 --image_size 256 512 --max_images 5 10 15
```

```bash
nohup python bulk_pipeline.py \
  --n‐estimators 50 100 200 \
  --max‐depth 10 20 \
  --feature‐limit 50 100 \
  --n‐jobs -1 \
  > sweep.log 2>&1 &
```