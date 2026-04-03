# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss, RMSELoss, ScaleAndShiftInvariantLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.ssi_loss = ScaleAndShiftInvariantLoss()
        self.grad_loss = GradL1Loss()
        self.rmse_loss = RMSELoss()
        self.domain_classifier_loss = nn.CrossEntropyLoss()

        self.scaler = amp.GradScaler(enabled=self.config.use_amp)

    def train_on_batch(self, batch, train_step):
        images, depths_gt, sds = batch['image'].to(
            self.device), batch['depth'].to(self.device), batch['sparse_depth'].to(self.device)

        dataset = batch['dataset'][0]
        domain_labels = torch.Tensor([dataset == 'kitti' for _ in range(images.size(0))]).to(torch.long).to(self.device)

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}
        with amp.autocast(enabled=self.config.use_amp):
            output = self.model(images, sds)
            pred_depths = output['metric_depth']
            domain_logits = output['domain_logits']

            if pred_depths.numel() > 1 or domain_logits.numel() > 1:
                #把nan值置0
                pred_depths = torch.nan_to_num(pred_depths, nan=0.001)
                depths_gt = torch.nan_to_num(depths_gt, nan=0.001)

                # print("pred_depths",pred_depths)
                # print("depths_gt",depths_gt)
                l_si, pred = self.silog_loss(pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
                # print("l_si",l_si)
                # print("pred",pred)
                loss = self.config.w_si * l_si
                losses[self.silog_loss.name] = l_si

                if self.config.w_rmse > 0:
                    l_rmse = self.rmse_loss(pred, depths_gt, mask=mask)
                    loss += self.config.w_rmse * l_rmse
                    losses[self.rmse_loss.name] = l_rmse
                else:
                    l_rmse = torch.Tensor([0])

                # 打印损失值
                # print(f"Silog Loss: {l_si.item()}, RMSE Loss: {l_rmse.item() if self.config.w_rmse > 0 else 0}")

            else:
                loss = loss 

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and self.step > 1 and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return losses

    def validate_on_batch(self, batch, val_step):
        # images = batch['image'].to(self.device)
        # depths_gt = batch['depth'].to(self.device)
        images, depths_gt, sds= batch['image'].to(
            self.device), batch['depth'].to(self.device), batch['sparse_depth'].to(self.device)
        dataset = batch['dataset'][0]
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        with amp.autocast(enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            pred_depths = m(images,sds)["metric_depth"]
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

        mask = torch.logical_and(
            depths_gt > self.config.min_depth, depths_gt < self.config.max_depth)
        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)

        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        return metrics, losses