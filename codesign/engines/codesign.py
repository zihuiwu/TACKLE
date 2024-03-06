import wandb, torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from codesign.utils.wandb_imshow import wandb_imshow

def imshow(img, caption):
    fig = plt.figure()
    plt.imshow(img[0].detach().cpu(), 'gray')
    plt.colorbar()
    wandb.log({caption: fig})
    plt.close()

class LitCoDesign(pl.LightningModule):
    def __init__(self, cfg, sampler, reconstructor, predictor, train_loss, val_test_loss):
        super().__init__()
        self.cfg = cfg
        self.sampler = sampler
        self.reconstructor = reconstructor
        self.predictor = predictor
        self.train_loss = train_loss
        self.val_test_loss = val_test_loss
        self.save_hyperparameters(ignore=['sampler', 'reconstructor', 'predictor', 'train_loss', 'val_test_loss'])

    @property
    def name(self):
        return self.cfg.model.name

    @property
    def task(self):
        return self.cfg.task

    def _calc_loss_by_task(self, recon, pred, image, segmap, bbox, label):
        args_dict = {
            'recon': (pred, image),
            'local_recon': (pred, image, bbox),
            'local_enhance': (recon, pred, image, bbox),
            'seg': (pred, segmap),
            'seg_reg': (pred, segmap, recon, image),
            'cls': (pred, label)
        }
        train_args = args_dict[self.train_loss.task]
        val_test_args = args_dict[self.val_test_loss.task]
        train_loss = self.train_loss(*train_args)
        val_or_test_loss = self.val_test_loss(*val_test_args)
        return train_loss, val_or_test_loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        kspace, image, segmap, bbox, label = batch
        kspace_sampled, mask_binarized = self.sampler(kspace)
        recon, _ = self.reconstructor(kspace_sampled, mask_binarized)
        pred = self.predictor(recon)
        pred_train_loss, pred_val_test_loss = self._calc_loss_by_task(recon, pred, image, segmap, bbox, label)
        return {
            'loss': pred_train_loss,
            'val_test_loss': pred_val_test_loss
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # log and visualize
        pred_train_loss, pred_val_test_loss = outputs['loss'], outputs['val_test_loss']
        accuracy_dict = {
            f'train_{self.train_loss.name}': pred_train_loss.item(),
            f'train_{self.val_test_loss.name}': pred_val_test_loss.item()
        }
        self.log_dict(accuracy_dict, on_step=False, on_epoch=True) # change the default on_step, on_epoch options for "on_train_batch_end" hook
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        kspace, image, segmap, bbox, label = batch
        kspace_sampled, mask_binarized = self.sampler(kspace)
        recon, _ = self.reconstructor(kspace_sampled, mask_binarized)
        pred = self.predictor(recon)
        pred_train_loss, pred_val_loss = self._calc_loss_by_task(recon, pred, image, segmap, bbox, label)
        return pred_train_loss, pred_val_loss

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # log and visualize
        pred_train_loss, pred_val_loss = outputs
        accuracy_dict = {
            f'val_{self.train_loss.name}': pred_train_loss.item(),
            f'val_{self.val_test_loss.name}': pred_val_loss.item()
        }
        self.log_dict(accuracy_dict)
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            try:
                wandb_imshow(self.sampler.mask_binarized_vis, 'binary mask')
            except:
                # no wandb logger found
                pass

    def test_step(self, batch, batch_idx):
        # this is the test loop
        kspace, image, segmap, bbox, label = batch
        kspace_sampled, mask_binarized = self.sampler(kspace)
        recon, recon_zf = self.reconstructor(kspace_sampled, mask_binarized)
        pred = self.predictor(recon)
        pred_train_loss, pred_test_loss = self._calc_loss_by_task(recon, pred, image, segmap, bbox, label)
        if self.task in ['recon', 'local_recon', 'local_enhance']:
            zf_train_loss, zf_test_loss = self._calc_loss_by_task(recon, recon_zf, image, segmap, bbox, label)
        else:
            zf_train_loss, zf_test_loss = None, None
        return pred_train_loss, pred_test_loss, zf_train_loss, zf_test_loss, kspace_sampled, mask_binarized, recon, recon_zf, pred

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # log and visualize
        pred_train_loss, pred_test_loss, zf_train_loss, zf_test_loss, _, _, _, _, _ = outputs
        accuracy_dict = {
            f'test_{self.train_loss.name}': pred_train_loss.item(),
            f'test_{self.val_test_loss.name}': pred_test_loss.item()
        }
        if self.task in ['recon', 'local_recon', 'local_enhance']:
            accuracy_dict.update({
                f'test_zf_{self.train_loss.name}': zf_train_loss.item(),
                f'test_zf_{self.val_test_loss.name}': zf_test_loss.item()
            })
        self.log_dict(accuracy_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer


class LitCrossValidationCoDesign(LitCoDesign):
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log and visualize
        current_fold = self.trainer.fit_loop.current_fold
        pred_train_loss, pred_val_loss = outputs
        accuracy_dict = {
            f'val_{self.train_loss.name}_fold{current_fold}': pred_train_loss.item(),
            f'val_{self.val_test_loss.name}_fold{current_fold}': pred_val_loss.item()
        }
        self.log_dict(accuracy_dict)
        if self.current_epoch % 10 == 0 and batch_idx == 0:
            fig = plt.figure()
            plt.imshow(self.sampler.mask_binarized_vis, 'gray')
            plt.colorbar()
            wandb.log({f'binary mask (fold {current_fold})': fig})
            plt.close()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        current_fold = self.trainer.fit_loop.current_fold
        pred_train_loss, pred_test_loss, zf_train_loss, zf_test_loss, _, _, _, _, _ = outputs
        accuracy_dict = {
            f'test_{self.train_loss.name}_fold{current_fold}': pred_train_loss.item(),
            f'test_{self.val_test_loss.name}_fold{current_fold}': pred_test_loss.item()
        }
        if self.task in ['recon', 'local_recon', 'local_enhance']:
            accuracy_dict.update({
                f'test_zf_{self.train_loss.name}_fold{current_fold}': zf_train_loss.item(),
                f'test_zf_{self.val_test_loss.name}_fold{current_fold}': zf_test_loss.item()
            })
        self.log_dict(accuracy_dict)