# Ray Distributed Training

Distributed ML training and hyperparameter optimization using Ray Train and Ray Tune.

## Features
- **Ray Train**: Multi-GPU distributed training with `TorchTrainer`
- **ASHA Scheduler**: Async Successive Halving for early stopping of bad trials
- **Population Based Training (PBT)**: Evolutionary hyperparameter optimization
- **Optuna Search**: Bayesian search algorithm integration
- **Checkpointing**: Automatic best-N checkpoint saving

## Usage
```python
from src.ray_trainer import build_trainer
from src.ray_tune_search import run_asha_search, EXAMPLE_PARAM_SPACE

# Distributed training (4 GPUs)
trainer = build_trainer(model_fn, num_workers=4, use_gpu=True)
result = trainer.fit()

# Hyperparameter search
results = run_asha_search(model_fn, EXAMPLE_PARAM_SPACE, n_samples=50)
best = results.get_best_result(metric='accuracy', mode='max')
print(best.config)
```
