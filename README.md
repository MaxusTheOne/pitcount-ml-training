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
python training_package\bulk_pipeline.py --n_estimators 10 --max_depth 20 None --feature_limit 0.1 0.2 0.3 sqrt log2 --n_jobs -1 --n_components 30 50 --image_size 512 1024 --max_images 15 --output_dir 'C:\Users\marku\OneDrive - Københavns Erhvervsakademi\Desktop\main_proj\models_bulk' --skip_existing 
```

```bash
nohup python bulk_pipeline.py \
  --n‐estimators 50 100 200 \
  --max‐depth 10 20 \
  --feature‐limit 50 100 \
  --n‐jobs -1 \
  > sweep.log 2>&1 &
```