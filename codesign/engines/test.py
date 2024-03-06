import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from codesign.engines.callbacks import TestCallback

class Test:
    def __init__(self, cfg, data_cfg) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg

    def test_run(self, args, model, data_module) -> None:
        # callbacks
        callbacks = []
        from codesign.config import CodesignTestConfigurator
        if 'callbacks' in self.data_cfg.keys():
            for callback_name in self.data_cfg.callbacks.values():
                test_callback_type = CodesignTestConfigurator.str_to_class(
                    "codesign.engines.callbacks", 
                    callback_name
                )
                callbacks.append(test_callback_type(type(data_module).__name__, self.cfg, self.data_cfg))
        else:
            callbacks.append(TestCallback(type(data_module).__name__, self.cfg, self.data_cfg))

        logger = WandbLogger(id=args.id, **dict(self.cfg.logger), entity='wu_yin') if args.id else False
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=1, 
            logger=logger,
            callbacks=callbacks
        )
        trainer.test(model, data_module)