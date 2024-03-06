import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.loops import Loop
from pytorch_lightning.trainer.states import TrainerFn
from codesign.data.base import BaseKFoldDataModule
from typing import Any, Dict
from copy import deepcopy

#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, cfg, num_folds: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_folds = num_folds
        self.current_fold: int = 0

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)
        for i, cb in enumerate(self.trainer.callbacks):
            if isinstance(cb, ModelCheckpoint):
                loss_display_name = self.trainer.model.val_test_loss.name
                self.trainer.callbacks[i] = ModelCheckpoint(
                    dirpath="results/fashion_mnist_recon_4x",
                    filename=f'fold{self.current_fold}_{{epoch}}_{{val_{loss_display_name}_fold{self.current_fold}:.2f}}',
                    monitor=f'val_{loss_display_name}_fold{self.current_fold}',
                    mode=self.trainer.model.val_test_loss.mode, 
                    verbose=False
                )
            elif isinstance(cb, EarlyStopping):
                self.trainer.callbacks[i] = EarlyStopping(
                    monitor=f'val_{loss_display_name}_fold{self.current_fold}',
                    patience=20, 
                    mode=self.trainer.model.val_test_loss.mode, 
                    verbose=False
                )

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        # self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        # checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        # voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        # voting_model.trainer = self.trainer
        # # This requires to connect the new model and move it the right device.
        # self.trainer.strategy.connect(voting_model)
        # self.trainer.strategy.model_to_device()
        # self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


class CrossValidation:
    def __init__(self, cfg, num_folds):
        self.cfg = cfg
        self.num_folds = num_folds
 
    def __call__(self, model, data_module):
        # train model
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            logger=WandbLogger(**dict(self.cfg.logger)), 
            **dict(self.cfg.trainer)
        )

        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = KFoldLoop(self.cfg, self.num_folds)
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model, data_module)