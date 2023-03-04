# %%
import datetime
import shutil
from pathlib import Path

import gin
import gin.torch.external_configurables # TODO: gin.torch doesn't actually import gin.torch.external_configurables --> PR?
import click
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .. import layers
from ..networks import MultiRBPNet
from ..losses import MultinomialNLLLossFromLogits
from ..metrics import MultinomialNLLFromLogits, BatchedPearsonCorrCoef
from ..data.datasets import TFIterableDataset
from ..data import tfrecord_to_dataloader, dummy_dataloader

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


# %%
@gin.configurable()
class Model(pl.LightningModule):
    def __init__(self, network, _example_input, metrics=None, optimizer=torch.optim.Adam):
        super().__init__()
        self.network = network
        self.loss_fn = MultinomialNLLLossFromLogits()
        
        # metrics
        if metrics is None:
            self.metrics = nn.ModuleDict({})
        else:
            self.metrics = nn.ModuleDict(metrics)
        
        # optimizer
        self.optimizer_cls = optimizer
    
    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, **kwargs):
        inputs, y = batch
        y = y['total']
        y_pred = self.forward(inputs)
        loss = self.loss_fn(y, y_pred, dim=-2)
        self.compute_and_log_metics(y_pred, y, partition='train')
        return loss
    
    def training_epoch_end(self, *args, **kwargs):
        self._reset_metrics()

    def validation_epoch_end(self, *args, **kwargs):
        self._reset_metrics()

    def validation_step(self, batch, batch_idx):
        inputs, y = batch
        y = y['total']
        y_pred = self.forward(inputs)
        self.compute_and_log_metics(y_pred, y, partition='val')
    
    def compute_and_log_metics(self, y_pred, y, partition=None):
        on_step = False
        if partition == 'train':
            on_step = True

        for name, metric in self.metrics.items():
            metric(y_pred, y)
            self.log(f'{partition}/{name}', metric.compute(), on_step=on_step, on_epoch=True, prog_bar=False)
    
    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

# %%
def _make_callbacks(output_path):
    callbacks = [
        ModelCheckpoint(dirpath=output_path/'checkpoints', every_n_epochs=1, save_last=True),
        EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode='min'),
    ]
    return callbacks

# %%
def _make_loggers(output_path):
    loggers = [
        pl_loggers.TensorBoardLogger(output_path/'tensorboard', name='', version='', log_graph=True),
    ]
    return loggers

# %%
@gin.configurable(denylist=['tfrecord', 'validation_tfrecord', 'output_path'])
def train(tfrecord, validation_tfrecord, output_path, dataset=TFIterableDataset, metrics=None, optimizer=None, batch_size=128, shuffle=None, network=None, **kwargs):
    dataloader_train = torch.utils.data.DataLoader(dataset(filepath=tfrecord, batch_size=batch_size, shuffle=shuffle), batch_size=None) #tfrecord_to_dataloader(tfrecord, batch_size=batch_size, shuffle=shuffle)
    if validation_tfrecord is not None:
        dataloader_val = torch.utils.data.DataLoader(dataset(filepath=validation_tfrecord, batch_size=batch_size, shuffle=shuffle), batch_size=None)
    else:
        dataloader_val = None

    trainer = pl.Trainer(
        default_root_dir=output_path, 
        logger=_make_loggers(output_path), 
        callbacks=_make_callbacks(output_path),
        **kwargs,
        )

    model = Model(network, next(iter(dataloader_train))[0], metrics=metrics, optimizer=optimizer)
    
    # write model summary
    with open(str(output_path / 'model.summary.txt'), 'w') as f:
        print(str(model), file=f)
    
    # write optimizer summary
    with open(str(output_path / 'optim.summary.txt'), 'w') as f:
        print(str(model.configure_optimizers()), file=f)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    # torch.save(model, output_path / 'model.pt') # Raises Tensorflow error during pickling (InvalidArgumentError: Cannot convert a Tensor of dtype variant to a NumPy array.)
    torch.save(model.network, output_path / 'network.pt')

    print('Final validation ...')
    with open(str(output_path.parent / 'result'), 'w') as f:
        result = trainer.validate(model, dataloader_val)[0]['val/loss_epoch']
        print(result, file=f)


# %%
@click.command()
@click.argument('tfrecord', required=True, type=str)
@click.option('--config', type=str, default=None)
@click.option('-o', '--output', required=True)
@click.option('--validation-tfrecord', type=str, default=None)
def main(tfrecord, config, output, validation_tfrecord):
    # parse gin-config config
    if config is not None:
        gin.parse_config_file(config)

    output_path = Path(f'{output}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(parents=True)
    if config is not None:
        shutil.copy(config, str(output_path / 'config.gin'))   

    train(tfrecord, validation_tfrecord, output_path)