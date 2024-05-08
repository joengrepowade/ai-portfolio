import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable


def training_loop(config: Dict[str, Any]):
    """Ray Train distributed training loop."""
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']

    model = config['model_fn']()
    model = prepare_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=config.get('weight_decay', 0.01)
    )
    criterion = nn.CrossEntropyLoss()

    train_ds = train.get_dataset_shard('train')
    val_ds = train.get_dataset_shard('val')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        epochs=epochs,
        steps_per_epoch=config.get('steps_per_epoch', 100)
    )

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_ds.iter_torch_batches(batch_size=batch_size):
            inputs = batch['input'].float()
            labels = batch['label'].long()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train.report({
            'epoch': epoch + 1,
            'train_loss': total_loss,
            'train_acc': correct / max(total, 1),
            'lr': scheduler.get_last_lr()[0]
        })


def build_trainer(model_fn: Callable, num_workers: int = 4,
                  use_gpu: bool = True, storage_path: str = '/tmp/ray_results') -> TorchTrainer:
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={'GPU': 1, 'CPU': 4} if use_gpu else {'CPU': 4}
    )
    run_config = RunConfig(
        storage_path=storage_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute='train_acc',
            checkpoint_score_order='max'
        )
    )
    return TorchTrainer(
        train_loop_per_worker=training_loop,
        scaling_config=scaling_config,
        run_config=run_config,
    )
