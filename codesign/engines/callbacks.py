import torch, os, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from pytorch_lightning.callbacks import Callback
from codesign.utils.fftn import fftn_
from codesign.utils.ifftn import ifftn_
from codesign.utils.imsave import imsave
from codesign.utils.seg_argmax_pred import seg_argmax_pred
from codesign.utils.plot_confusion_matrix import plot_confusion_matrix
from codesign.losses.psnr_loss import PSNR
from codesign.losses.psnr_local_loss import PSNRLocal
from codesign.losses.dice_monai_loss import DiceMonai
from sklearn.metrics import confusion_matrix

def get_roi(image: torch.Tensor, bbox: torch.Tensor):
    """_summary_

    Args:
        image (torch.Tensor): an image whose region of interest is to be cropped and visualized
        bbox (torch.Tensor): region of interest 
    """
    nz = torch.nonzero(bbox)  # indices of all nonzero elements
    top_left = nz.min(dim=0)[0]
    bottom_right = nz.max(dim=0)[0]
    roi = image[top_left[0]:bottom_right[0]+1,
                top_left[1]:bottom_right[1]+1]
    return roi

class TestCallback(Callback):
    def __init__(self, name, cfg, data_cfg) -> None:
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.vis_freq = data_cfg.vis_freq
        self.save_dir = f'{cfg.exp_dir}/test_{name}'
        os.makedirs(self.save_dir, exist_ok=True)

    def _recon_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, _, bbox, _ = batch
        _, pred_test_loss, _, zf_test_loss, kspace_sampled, mask_binarized, _, recon_zf, pred = outputs

        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        if batch_idx % self.vis_freq == 0:
            kspace_recon = fftn_(pred[0,0], dim=(0,1))
            if self.data_cfg.task == 'local_recon':
                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
            else:
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            sp1 = axes[0, 0].imshow(torch.log(kspace.abs())[0].detach().cpu(), 'gray')
            sp2 = axes[0, 1].imshow(torch.log(kspace_sampled.abs())[0].detach().cpu(), 'gray')
            sp3 = axes[0, 2].imshow(torch.log(kspace_recon.abs()).detach().cpu(), 'gray')
            sp4 = axes[1, 0].imshow(image[0,0].detach().cpu(), 'gray')
            sp5 = axes[1, 1].imshow(recon_zf[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)
            sp6 = axes[1, 2].imshow(pred[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)
            if self.data_cfg.task == 'local_recon':
                axes[1, 0].contour(bbox[0,0].detach().cpu(), levels=1)
                axes[1, 1].contour(bbox[0,0].detach().cpu(), levels=1)
                axes[1, 2].contour(bbox[0,0].detach().cpu(), levels=1)
                image_roi = get_roi(image[0,0], bbox[0,0])
                recon_zf_roi = get_roi(recon_zf[0,0], bbox[0,0])
                pred_roi = get_roi(pred[0,0], bbox[0,0])
                image_roi_min, image_roi_max = image_roi.min().item(), image_roi.max().item()
                sp7 = axes[2, 0].imshow(image_roi.detach().cpu(), 'gray')
                sp8 = axes[2, 1].imshow(recon_zf_roi.detach().cpu(), 'gray', vmin=image_roi_min, vmax=image_roi_max)
                sp9 = axes[2, 2].imshow(pred_roi.detach().cpu(), 'gray', vmin=image_roi_min, vmax=image_roi_max)
                axes[2, 0].set_title(f'ground truth RoI')
                zf_recon_roi_test_loss_str = f' ({pl_module.val_test_loss.name}={zf_test_loss.item():.2f})' if self.data_cfg.data_module.name != 'MartinosProspectiveDataModule' else ''
                recon_roi_test_loss_str = f' ({pl_module.val_test_loss.name}={pred_test_loss.item():.2f})' if self.data_cfg.data_module.name != 'MartinosProspectiveDataModule' else ''
                axes[2, 1].set_title(f'zero-filled recon RoI' + zf_recon_roi_test_loss_str)
                axes[2, 2].set_title(f'recon RoI' + recon_roi_test_loss_str)
                fig.colorbar(sp7, ax=axes[2, 0])
                fig.colorbar(sp8, ax=axes[2, 1])
                fig.colorbar(sp9, ax=axes[2, 2])
            
            axes[0, 0].set_title(f'k-space')
            axes[0, 1].set_title(f'subsampled k-space')
            axes[0, 2].set_title(f'k-space of recon.')
            axes[1, 0].set_title(f'ground truth')
            zf_recon_test_loss_str = f' ({pl_module.val_test_loss.name}={zf_test_loss.item():.2f})' if self.data_cfg.data_module.name != 'MartinosProspectiveDataModule' else ''
            recon_test_loss_str = f' ({pl_module.val_test_loss.name}={pred_test_loss.item():.2f})' if self.data_cfg.data_module.name != 'MartinosProspectiveDataModule' else ''
            axes[1, 1].set_title(f'zero-filled recon' + zf_recon_test_loss_str)
            axes[1, 2].set_title(f'recon' + recon_test_loss_str)
            fig.colorbar(sp1, ax=axes[0, 0])
            fig.colorbar(sp2, ax=axes[0, 1])
            fig.colorbar(sp3, ax=axes[0, 2])
            fig.colorbar(sp4, ax=axes[1, 0])
            fig.colorbar(sp5, ax=axes[1, 1])
            fig.colorbar(sp6, ax=axes[1, 2])
            if 'Martinos' in self.name:
                fig.suptitle(f'Readout Slice {batch_idx}')
            else:
                fig.suptitle(f'Test Sample {batch_idx}')
            plt.savefig(f'{self.save_dir}/test_{batch_idx}.png')
            plt.tight_layout()
            plt.close()

    def _get_compound_segmap(self, segmap: torch.Tensor):
        """_summary_

        Args:
            segmap (torch.Tensor): an multi-class segmap of size (C*H*W) where each channel contains the segmap of one class
        """
        assert segmap.dim() == 3, 'The segmap must have shape C*H*W!'
        
        compound_segmap = torch.zeros(*segmap.shape[1:]).to(segmap.device)
        for i in range(len(segmap)):
            compound_segmap += i * segmap[i]
            compound_segmap_new = torch.clamp(compound_segmap, max=i) 
            if torch.any(compound_segmap_new != compound_segmap):
                warnings.warn('There are pixels predicted to be in more than 1 class!!')
            compound_segmap = compound_segmap_new

        return compound_segmap

    def _seg_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, segmap, _, _ = batch
        _, pred_test_loss, _, zf_test_loss, kspace_sampled, mask_binarized, recon, recon_zf, pred = outputs

        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        if batch_idx % self.vis_freq == 0:
            kspace_recon = fftn_(recon[0,0], dim=(0,1))
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15), dpi=50)
            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            sp1 = axes[0, 0].imshow(torch.log(kspace.abs())[0].detach().cpu(), 'gray')
            sp2 = axes[0, 1].imshow(torch.log(kspace_sampled.abs())[0].detach().cpu(), 'gray')
            sp3 = axes[0, 2].imshow(torch.log(kspace_recon.abs()).detach().cpu(), 'gray')
            sp4 = axes[1, 0].imshow(image[0,0].detach().cpu(), 'gray')
            sp5 = axes[1, 1].imshow(recon_zf[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)
            sp6 = axes[1, 2].imshow(recon[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)

            pred = seg_argmax_pred(pred, chan_dim=1)
            compound_segmap = self._get_compound_segmap(segmap[0])
            compound_pred = self._get_compound_segmap(pred[0])
            error = torch.clamp(torch.abs(compound_segmap - compound_pred), max=1)
            sp7 = axes[2, 0].imshow(compound_segmap.detach().cpu())
            sp8 = axes[2, 1].imshow(error.detach().cpu())
            sp9 = axes[2, 2].imshow(compound_pred.detach().cpu())

            fig.colorbar(sp7, ax=axes[2, 0])
            fig.colorbar(sp8, ax=axes[2, 1])
            fig.colorbar(sp9, ax=axes[2, 2])
            axes[2, 0].set_title(f'ground truth')
            axes[2, 1].set_title(f'error')
            axes[2, 2].set_title(f'prediction ({pl_module.val_test_loss.name}={pred_test_loss.item():.4f})')
            
            axes[0, 0].set_title(f'k-space')
            axes[0, 1].set_title(f'subsampled k-space')
            axes[0, 2].set_title(f'k-space of recon.')
            axes[1, 0].set_title(f'ground truth')
            axes[1, 1].set_title(f'zero-filled recon')
            axes[1, 2].set_title(f'recon')
            fig.colorbar(sp1, ax=axes[0, 0])
            fig.colorbar(sp2, ax=axes[0, 1])
            fig.colorbar(sp3, ax=axes[0, 2])
            fig.colorbar(sp4, ax=axes[1, 0])
            fig.colorbar(sp5, ax=axes[1, 1])
            fig.colorbar(sp6, ax=axes[1, 2])
            if 'Martinos' in self.name:
                fig.suptitle(f'Readout Slice {batch_idx}')
            else:
                fig.suptitle(f'Test Sample {batch_idx}')
            plt.savefig(f'{self.save_dir}/test_{batch_idx}.png')
            plt.tight_layout()
            plt.close()

    def _seg_for_cls_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        _, _, _, _, label = batch
        _, _, _, _, _, _, _, _, pred = outputs

        pred = seg_argmax_pred(pred, chan_dim=1)
        compound_pred = self._get_compound_segmap(pred[0])
        if torch.any(compound_pred!=0):
            cls_pred_from_seg = torch.Tensor([[0, 1]])
        else:
            cls_pred_from_seg = torch.Tensor([[1, 0]])

        self.preds.append(cls_pred_from_seg)
        self.labels.append(label.detach().cpu())
        self.pred_sums.append(torch.sum(compound_pred).item())

        self._seg_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _cls_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, _, _, label = batch
        _, _, _, _, kspace_sampled, _, recon, recon_zf, pred = outputs

        self.preds.append(pred.detach().cpu())
        self.labels.append(label.detach().cpu())

        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        if batch_idx % self.vis_freq == 0:
            kspace_recon = fftn_(recon[0,0], dim=(0,1))
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), dpi=50)
            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            sp1 = axes[0, 0].imshow(torch.log(kspace.abs())[0].detach().cpu(), 'gray')
            sp2 = axes[0, 1].imshow(torch.log(kspace_sampled.abs())[0].detach().cpu(), 'gray')
            sp3 = axes[0, 2].imshow(torch.log(kspace_recon.abs()).detach().cpu(), 'gray')
            sp4 = axes[1, 0].imshow(image[0,0].detach().cpu(), 'gray')
            sp5 = axes[1, 1].imshow(recon_zf[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)
            sp6 = axes[1, 2].imshow(recon[0,0].detach().cpu(), 'gray', vmin=image_min, vmax=image_max)
            
            axes[0, 0].set_title(f'k-space')
            axes[0, 1].set_title(f'subsampled k-space')
            axes[0, 2].set_title(f'k-space of recon.')
            axes[1, 0].set_title(f'ground truth')
            axes[1, 1].set_title(f'zero-filled recon')
            axes[1, 2].set_title(f'recon')
            fig.colorbar(sp1, ax=axes[0, 0])
            fig.colorbar(sp2, ax=axes[0, 1])
            fig.colorbar(sp3, ax=axes[0, 2])
            fig.colorbar(sp4, ax=axes[1, 0])
            fig.colorbar(sp5, ax=axes[1, 1])
            fig.colorbar(sp6, ax=axes[1, 2])
            if 'Martinos' in self.name:
                fig.suptitle(f'Readout Slice {batch_idx}')
            else:
                fig.suptitle(f'Test Sample {batch_idx}')
            plt.savefig(f'{self.save_dir}/test_{batch_idx}.png')
            plt.tight_layout()
            plt.close()

    def on_test_epoch_start(self, trainer, pl_module):
        if self.data_cfg.task in ['cls', 'seg_for_cls']:
            self.preds = []
            self.labels = []
            self.pred_sums = []

    def on_test_epoch_end(self, trainer, pl_module):
        with open(f'{self.save_dir}/metrics.txt', 'w') as f:
            for k, v in trainer.logged_metrics.items():
                f.write(f'{k}: {v.item()}\n')
        
        if self.data_cfg.task in ['cls', 'seg_for_cls']:
            f1_score, acc = plot_confusion_matrix(
                cm=confusion_matrix(y_true=torch.cat(self.labels, dim=0), y_pred=torch.cat(self.preds, dim=0).argmax(1)), 
                target_names=trainer.test_dataloaders[0].dataset.class_names.values(), 
                fname=f'{self.save_dir}/test_confusion_matrix.png',
                normalize=False
            )

            if self.data_cfg.task in ['seg_for_cls']:
                pl_module.log_dict({
                    f'test_f1_score': f1_score,
                    f'test_classification_accuracy': acc
                })
                savemat(
                    f'{self.save_dir}/pred_label_infos.mat', 
                    {
                        'pred_sums': self.pred_sums, 
                        'labels': torch.cat(self.labels, dim=0).detach().cpu().numpy(),
                    }
                )
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.data_cfg.task in ['recon', 'local_recon', 'local_enhance']:
            self._recon_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif self.data_cfg.task in ['seg']:
            self._seg_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif self.data_cfg.task in ['seg_for_cls']:
            self._seg_for_cls_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif self.data_cfg.task in ['cls']:
            self._cls_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class PaperFigureCallback(Callback):
    def __init__(self, name, cfg, data_cfg) -> None:
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.vis_freq = data_cfg.vis_freq
        self.save_dir = f'{cfg.exp_dir}/paper_figure_{name}'
        os.makedirs(self.save_dir, exist_ok=True)

    def _recon_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, _, bbox, _ = batch
        _, pred_test_loss, _, zf_test_loss, kspace_sampled, mask_binarized, _, recon_zf, pred = outputs

        self.test_loss_list.append(pred_test_loss.item())
        
        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        extra_condition = False if 'paper_figure_vis_list' not in self.data_cfg.keys() else batch_idx in self.data_cfg.paper_figure_vis_list
        if batch_idx % self.vis_freq == 0 or extra_condition:
            example_dir = f'{self.save_dir}/test_{batch_idx}'
            os.makedirs(example_dir, exist_ok=True)
            kspace_recon = fftn_(pred[0,0], dim=(0,1))
            psf = ifftn_(mask_binarized[0], dim=(0,1))
            savemat(f'{example_dir}/psf.mat', {'psf': psf.detach().cpu().numpy(), 'psf_log': torch.log(psf.abs()).detach().cpu().numpy()})
            
            data_for_saving = {
                'gt': image[0,0].detach().cpu().numpy(),
                'recon': pred[0,0].detach().cpu().numpy(), 
                'recon_zf': recon_zf[0,0].detach().cpu().numpy(), 
            }

            psnr = PSNR()
            psnr_zf = psnr(recon_zf, image)
            psnr_recon = psnr(pred, image)
            
            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            plt.imsave(f'{example_dir}/kspace_coil=0.png', self.inf_to_num_(torch.log(kspace.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/kspace_sampled_coil=0.png', self.inf_to_num_(torch.log(kspace_sampled.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/mask.png', mask_binarized[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf.png', psf.abs().detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf_log.png', torch.log(psf.abs()).detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/kspace_recon.png', torch.log(kspace_recon.abs()).detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/gt.png', image[0,0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/recon_zf_{100*psnr_zf:.0f}.png', recon_zf[0,0].detach().cpu(), cmap='gray', vmin=image_min, vmax=image_max)
            plt.imsave(f'{example_dir}/recon_{100*psnr_recon:.0f}.png', pred[0,0].detach().cpu(), cmap='gray', vmin=image_min, vmax=image_max)
            plt.imsave(f'{example_dir}/error_recon.png', torch.abs(image[0,0]-pred[0,0]).detach().cpu(), cmap='gray')
            if self.data_cfg.task == 'local_recon':
                local_psnr = PSNRLocal()
                local_psnr_zf = local_psnr(recon_zf, image, bbox)
                local_psnr_recon = local_psnr(pred, image, bbox)

                image_roi = get_roi(image[0,0], bbox[0,0])
                recon_zf_roi = get_roi(recon_zf[0,0], bbox[0,0])
                pred_roi = get_roi(pred[0,0], bbox[0,0])
                image_roi_min, image_roi_max = image_roi.min().item(), image_roi.max().item()

                data_for_saving.update({
                    'roi': bbox[0,0].detach().cpu().numpy(),
                    'roi_gt': image_roi.detach().cpu().numpy(),
                    'roi_recon': pred_roi.detach().cpu().numpy(), 
                    'roi_recon_zf': recon_zf_roi.detach().cpu().numpy(), 
                })

                plt.imsave(f'{example_dir}/roi.png', bbox[0,0].detach().cpu(), cmap='gray')
                plt.imsave(f'{example_dir}/roi_gt.png', image_roi.detach().cpu(), cmap='gray')
                plt.imsave(f'{example_dir}/roi_recon_zf_{100*local_psnr_zf:.0f}.png', recon_zf_roi.detach().cpu(), cmap='gray', vmin=image_roi_min, vmax=image_roi_max)
                plt.imsave(f'{example_dir}/roi_recon_{100*local_psnr_recon:.0f}.png', pred_roi.detach().cpu(), cmap='gray', vmin=image_roi_min, vmax=image_roi_max)
                plt.imsave(f'{example_dir}/error_roi_recon.png', torch.abs(image_roi-pred_roi).detach().cpu(), cmap='gray')

                plt.figure()
                plt.imshow(image[0,0].detach().cpu(), 'gray')
                plt.contour(bbox[0,0].detach().cpu(), levels=1)
                plt.savefig(f'{example_dir}/gt_with_roi.png')
                plt.close()

                enlarge = True
                if enlarge:
                    def enlarge_roi(bbox: torch.Tensor):
                        larger_bbox = torch.zeros_like(bbox)
                        nz = torch.nonzero(bbox)  # indices of all nonzero elements
                        top_left = nz.min(dim=0)[0]
                        bottom_right = nz.max(dim=0)[0]

                        top_left[0] -= 0
                        bottom_right[0] += 2
                        top_left[1] -= 6
                        bottom_right[1] += 6

                        larger_bbox[top_left[0]:bottom_right[0]+1,
                                    top_left[1]:bottom_right[1]+1] = 1

                        return larger_bbox

                    bbox[0,0] = enlarge_roi(bbox[0,0])
                    image_roi = get_roi(image[0,0], bbox[0,0])
                    recon_zf_roi = get_roi(recon_zf[0,0], bbox[0,0])
                    pred_roi = get_roi(pred[0,0], bbox[0,0])
                    image_roi_min, image_roi_max = image_roi.min().item(), image_roi.max().item()

                    data_for_saving.update({
                        'larger_roi': bbox[0,0].detach().cpu().numpy(),
                        'larger_roi_gt': image_roi.detach().cpu().numpy(),
                        'larger_roi_recon': pred_roi.detach().cpu().numpy(), 
                        'larger_roi_recon_zf': recon_zf_roi.detach().cpu().numpy(), 
                    })

                    plt.imsave(f'{example_dir}/larger_roi.png', bbox[0,0].detach().cpu(), cmap='gray')
                    plt.imsave(f'{example_dir}/larger_roi_gt.png', image_roi.detach().cpu(), cmap='gray')
                    plt.imsave(f'{example_dir}/larger_roi_recon_zf.png', recon_zf_roi.detach().cpu(), cmap='gray', vmin=image_roi_min, vmax=image_roi_max)
                    plt.imsave(f'{example_dir}/larger_roi_recon.png', pred_roi.detach().cpu(), cmap='gray', vmin=image_roi_min, vmax=image_roi_max)

            savemat(f'{example_dir}/data.mat', data_for_saving)

    @staticmethod
    def inf_to_num_(arr, val):
        arr[torch.isinf(arr)] = val
        return arr

    def _get_vis_map(self, img_shape, seg_map, is_error_map=False):
        if is_error_map:
            start_idx = 0
            colors = ['#000000', '#FF0000', '#FF0000', '#FF0000', '#FF0000'] # first is background
        else:
            start_idx = 1
            # colors = ['#000000', '#FF6C0C', '#73A950', '#F54D80', '#F1D384'] # first is background
            colors = ['#000000', '#FF6C0C', '#73A950', '#F1D384', '#F54D80'] # first is background
        
        compound_seg_map_shape = tuple(img_shape + [4])
        compound_seg_map_color = torch.zeros(compound_seg_map_shape, dtype=torch.float32).to(seg_map.device)
        for i in range(start_idx, len(seg_map)):
            sm = seg_map[i, :, :].unsqueeze(-1)
            color = colors[i]
            r, g, b = [int(color[j:j+2], 16) for j in (1, 3, 5)]
            seg_map_color_shape = tuple(img_shape + [4])
            seg_map_color = torch.zeros(seg_map_color_shape, dtype=torch.float32).to(seg_map.device)
            seg_map_color[:, :, 0] = r
            seg_map_color[:, :, 1] = g
            seg_map_color[:, :, 2] = b
            seg_map_color[:, :, 3] = 255
            compound_seg_map_color += seg_map_color * sm
            torch.clamp_(compound_seg_map_color, min=0, max=255)

        return compound_seg_map_color

    def _seg_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, segmap, _, _ = batch
        _, pred_test_loss, _, zf_test_loss, kspace_sampled, mask_binarized, recon, recon_zf, pred = outputs
        _, _, *img_shape = pred.shape

        self.test_loss_list.append(pred_test_loss.item())

        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        extra_condition = False if 'paper_figure_vis_list' not in self.data_cfg.keys() else batch_idx in self.data_cfg.paper_figure_vis_list
        if batch_idx % self.vis_freq == 0 or extra_condition:
            example_dir = f'{self.save_dir}/test_{batch_idx}'
            os.makedirs(example_dir, exist_ok=True)
            kspace_recon = fftn_(pred[0,0], dim=(0,1))
            psf = ifftn_(mask_binarized[0], dim=(0,1))
            savemat(f'{example_dir}/psf.mat', {'psf': psf.detach().cpu().numpy(), 'psf_log': torch.log(psf.abs()).detach().cpu().numpy()})

            psnr = PSNR()
            psnr_zf = psnr(recon_zf, image)
            psnr_recon = psnr(recon, image)

            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            plt.imsave(f'{example_dir}/kspace_coil=0.png', self.inf_to_num_(torch.log(kspace.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/kspace_sampled_coil=0.png', self.inf_to_num_(torch.log(kspace_sampled.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/mask.png', mask_binarized[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf.png', psf.abs().detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf_log.png', torch.log(psf.abs()).detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/kspace_recon.png', torch.log(kspace_recon.abs()).detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/gt.png', image[0,0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/recon_zf_{100*psnr_zf:.0f}.png', recon_zf[0,0].detach().cpu(), cmap='gray', vmin=image_min, vmax=image_max)
            plt.imsave(f'{example_dir}/recon_{100*psnr_recon:.0f}.png', recon[0,0].detach().cpu(), cmap='gray', vmin=image_min, vmax=image_max)

            dice_monai = DiceMonai()
            dice_monai_pred = dice_monai(pred, segmap)[0]

            pred = seg_argmax_pred(pred, chan_dim=1)
            segmap_vis = self._get_vis_map(img_shape, segmap[0])
            pred_vis = self._get_vis_map(img_shape, pred[0])
            error_vis = self._get_vis_map(img_shape, segmap[0].type(torch.int64) != pred[0], is_error_map=True)
            imsave(f'{example_dir}/segmap_gt.png', segmap_vis.detach().cpu())
            imsave(f'{example_dir}/segmap_error.png', error_vis.detach().cpu())
            imsave(f'{example_dir}/segmap_pred_{10000*dice_monai_pred:.0f}.png', pred_vis.detach().cpu())

    def _cls_visualization(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        kspace, image, segmap, _, _ = batch
        _, pred_test_loss, _, zf_test_loss, kspace_sampled, mask_binarized, recon, recon_zf, pred = outputs
        _, _, *img_shape = pred.shape

        self.test_loss_list.append(pred_test_loss.item())

        if kspace.dim() == 4:
            # multicoil case and visualize the kspace of the first coil
            # kspace: B * C * H * W
            # kspace_sampled: B * C * H * W
            kspace = kspace[:,0,:,:]
            kspace_sampled = kspace_sampled[:,0,:,:]

        extra_condition = False if 'paper_figure_vis_list' not in self.data_cfg.keys() else batch_idx in self.data_cfg.paper_figure_vis_list
        if batch_idx % self.vis_freq == 0 or extra_condition:
            example_dir = f'{self.save_dir}/test_{batch_idx}'
            os.makedirs(example_dir, exist_ok=True)
            psf = ifftn_(mask_binarized[0], dim=(0,1))
            savemat(f'{example_dir}/psf.mat', {'psf': psf.detach().cpu().numpy(), 'psf_log': torch.log(psf.abs()).detach().cpu().numpy()})

            psnr = PSNR()
            psnr_zf = psnr(recon_zf, image)
            psnr_recon = psnr(recon, image)

            image_min, image_max = image[0,0].min().item(), image[0,0].max().item()
            plt.imsave(f'{example_dir}/kspace_coil=0.png', self.inf_to_num_(torch.log(kspace.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/kspace_sampled_coil=0.png', self.inf_to_num_(torch.log(kspace_sampled.abs()), 0)[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/mask.png', mask_binarized[0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf.png', psf.abs().detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/psf_log.png', torch.log(psf.abs()).detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/gt.png', image[0,0].detach().cpu(), cmap='gray')
            plt.imsave(f'{example_dir}/recon_zf_{100*psnr_zf:.0f}.png', recon_zf[0,0].detach().cpu(), cmap='gray', vmin=image_min, vmax=image_max)
            plt.imsave(f'{example_dir}/recon_{100*psnr_recon:.0f}.png', recon[0,0].detach().cpu(), cmap='gray')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.data_cfg.task in ['recon', 'local_recon', 'local_enhance']:
            self._recon_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif self.data_cfg.task in ['seg', 'seg_for_cls']:
            self._seg_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        elif self.data_cfg.task in ['cls']:
            self._cls_visualization(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer, pl_module):
        self.test_loss_list = []

    def on_test_epoch_end(self, trainer, pl_module):
        savemat(f'{self.save_dir}/{self.cfg.exp_name}_test_loss_list.mat', {'test_loss_list': self.test_loss_list})