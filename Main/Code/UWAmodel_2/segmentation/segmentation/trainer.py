"""
The trainer class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
import logging
import torch
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return: accuracy score and evaluation results
    		(overall accuracy, mean accuracy, mean IoU, fwavacc)
    """

    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)).astype(np.float32)
        )

    mean_iou = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwavacc


import sys
# sys.path.append(r"D:\python_learning\7.28test\semantic-segmentation-pytorch-master\semantic-segmentation-pytorch-master\segmentation\tools")
from datetime import datetime
# from __future__ import absolute_import, division, print_function

from segmentation.tools.validation import *
from segmentation.tools.logger import *

try:
    from tqdm import tqdm
    from tqdm import trange
except ImportError:
    print("tqdm and trange not found, disabling progress bars")


    def tqdm(iter):
        return iter


    def trange(iter):
        return iter

TQDM_COLS = 80


def cross_entropy2d(input, target):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # input: (n*h*w, c)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0.0
    target = target[mask]

    func_loss = torch.nn.CrossEntropyLoss()
    loss = func_loss(input, target)

    return loss


class Trainer(object):

    def __init__(self, model, optimizer, logger, num_epochs, train_loader,
                 test_loader=None,
                 epoch=0,
                 log_batch_stride=30,
                 check_point_epoch_stride=1,
                 scheduler=None,
                 patience=100):
        """
        :param model: A network model to train.
        :param optimizer: A optimizer.
        :param logger: The logger for writing results to Tensorboard.
        :param num_epochs: iteration count.
        :param train_loader: pytorch's DataLoader
        :param test_loader: pytorch's DataLoader
        :param epoch: the start epoch number.
        :param log_batch_stride: it determines the step to write log in the batch loop.
        :param check_point_epoch_stride: it determines the step to save a model in the epoch loop.
        :param scheduler: optimizer scheduler for adjusting learning rate.
        """
        self.patience = patience
        self.cuda = torch.cuda.is_available()
        self.model = model
        self.optim = optimizer
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epoches = num_epochs
        self.check_point_step = check_point_epoch_stride
        self.log_batch_stride = log_batch_stride
        self.scheduler = scheduler

        self.epoch = epoch

    def train(self):
        print(self.scheduler,self.patience)
        epochs_no_improve = 0
        early_stop = False
        if not next(self.model.parameters()).is_cuda and self.cuda:
            raise ValueError("A model should be set via .cuda() before constructing optimizer.")
        
        model_save_path = '/root/autodl-tmp/save_models'
        os.makedirs(model_save_path, exist_ok=True)
        best_val_miou = 0.0001
        lr_values = []
        Avg_loss = []
        Lose_data = []
        Val_miou = []
        for epoch in trange(self.epoch, self.num_epoches,
                            position=0,
                            desc='Train', ncols=TQDM_COLS):
            # train
            avg_loss,lose_data, val_mean_iou = self._train_epoch()
            Avg_loss.append(avg_loss)
            Lose_data.append(lose_data)
            Val_miou.append(val_mean_iou)
            # step forward to reduce the learning rate in the optimizer.
            current_lr = self.optim.param_groups[0]['lr']
            print(current_lr)
            
            if self.scheduler:
                self.scheduler.step(lose_data)

            # torch.save(self.model, f'{model_save_path}/epoch_{round(val_mean_iou,3)}.pth')
           
            lr_values.append(current_lr)    
            
            if val_mean_iou > best_val_miou:
                best_val_miou = val_mean_iou
                torch.save(self.model, f'{model_save_path}/best_model.pth')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                # print(epochs_no_improve)
                if epochs_no_improve >= self.patience and abs(epoch - self.num_epoches) < 10:
                    print(f"epochs_no_improve={epochs_no_improve}, Early stopping triggered!")
                    early_stop = True
                    break

        if not early_stop:
            print("Training completed without early stopping.")


        # 创建图形并绘制三个数组
        x = np.linspace(0, 10, len(lr_values))
        plt.plot(x, lr_values, color='black', label='lr')
        # 添加图例
        plt.legend()
        # 添加标题和轴标签
        plt.title('Lr')
        
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.savefig(f'{model_save_path}/learning_rate_curve.png')

        plt.clf()
        x = range(len(Lose_data))
        plt.plot(x, Lose_data, color='red', label='val_loss')
        plt.plot(x, Val_miou, color='blue', label='val_miou')
                # 添加图例
        plt.legend()
        # 添加标题和轴标签
        plt.title('Valing')
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.savefig(f'{model_save_path}/val_training.png')

        plt.clf()
        x = range(len(Avg_loss))
        plt.plot(x, Avg_loss, color='green', label='train_loss')
        plt.legend()
        # 添加标题和轴标签
        plt.title('Training')
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.savefig(f'{model_save_path}/training_loss.png')


    def evaluate(self):

        num_batches = len(self.test_loader)
        self.model.eval()

        total_loss = 0.0  # 用于累积每个epoch的总损失
        MIOU = []
        with torch.no_grad():
            for n_batch, (sample_batched) in tqdm(enumerate(self.test_loader),
                                                  total=num_batches,
                                                  leave=False,
                                                  desc="Valid epoch={}".format(self.epoch),
                                                  ncols=TQDM_COLS):
                lose_data, acc, acc_cls, mean_iou = self._eval_batch(sample_batched, n_batch, num_batches)
                total_loss = total_loss + lose_data
                MIOU.append(mean_iou)
        aver_mean_iou = np.mean(np.array(MIOU))
        aver_total_loss = total_loss / num_batches
        print("Epoch {} - Val Average Loss: {:.4f} - Aver_mean_IOU: {:.4f}".format(self.epoch, aver_total_loss,aver_mean_iou))
        logging.info("Epoch {} - Val Average Loss: {:.4f} - Aver_mean_IOU: {:.4f}".format(self.epoch, aver_total_loss,aver_mean_iou))
        return aver_total_loss
    
    def _train_epoch(self):
        total_loss = 0.0  # 用于累积每个epoch的总损失
        num_batches = len(self.train_loader)

        if self.test_loader:
            dataloader_iterator = iter(self.test_loader)

        for n_batch, (sample_batched) in tqdm(enumerate(self.train_loader),
                                              total=num_batches,
                                              leave=False,
                                              desc="Train epoch={}".format(self.epoch),
                                              ncols=TQDM_COLS):
            self.model.train()
            data = sample_batched['image']
            target = sample_batched['annotation']

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            self.optim.zero_grad()
            
            

            torch.cuda.empty_cache()

            score = self.model(data)
            # print("score.shape:",score.shape)
            loss = cross_entropy2d(score, target)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            
            loss.backward()
            self.optim.step()
            
            if n_batch % self.log_batch_stride != 0:
                continue

            total_loss += loss_data

            self.logger.store_checkpoint_var('img_width', data.shape[3])
            self.logger.store_checkpoint_var('img_height', data.shape[2])

            self.model.img_width = data.shape[3]
            self.model.img_height = data.shape[2]

            # write logs to Tensorboard.
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iou, fwavacc = \
                label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])

            self.logger.log_train(loss, 'loss', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc, 'acc', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
            self.logger.log_train(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
            self.logger.log_train(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)

            # write result images when starting epoch.
            # if n_batch == 0:
            #     log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
            #     log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
            #     self.logger.log_images_train(log_img, self.epoch, n_batch, num_batches,
            #                                  nrows=data.shape[0])

            # if the trainer has the test loader, it evaluates the model using the test data.
            if self.test_loader:
                self.model.eval()
                with torch.no_grad():
                    try:
                        sample_batched = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.test_loader)
                        sample_batched = next(dataloader_iterator)

                    lose_data, acc, acc_cls, mean_iou = self._eval_batch(sample_batched, n_batch, num_batches)

        avg_loss = total_loss / num_batches  # 计算平均损失
        print("Epoch {} - Train Average Loss: {:.4f}".format(self.epoch, avg_loss))
        logging.info("Epoch {} - Train Average Loss: {:.4f}".format(self.epoch, avg_loss))
        logging.info("Epoch {} -Val: lose_data:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iou:{:.4f}".format(self.epoch, lose_data, acc, acc_cls, mean_iou))
        
        return avg_loss,lose_data,mean_iou

    def _eval_batch(self, sample_batched, n_batch, num_batches):

        data = sample_batched['image']
        target = sample_batched['annotation']

        if self.cuda:
            data, target = data.cuda(), target.cuda()
        torch.cuda.empty_cache()

        score = self.model(data)

        loss = cross_entropy2d(score, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()
        acc, acc_cls, mean_iou, fwavacc = \
            label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])
        # print(f"n_batch:{n_batch}, num_batches:{num_batches}, loss:{loss}, acc:{acc}, mean_iou:{mean_iou}")
        self.logger.log_test(loss, 'loss', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc, 'acc', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
        self.logger.log_test(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
        self.logger.log_test(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)

        # if n_batch == 0:
        #     log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        #     log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
        #     self.logger.log_images_test(log_img, self.epoch, n_batch, num_batches,
        #                                 nrows=data.shape[0])

        return loss_data, acc, acc_cls, mean_iou

    def _write_img(self, score, target, input_img, n_batch, num_batches):
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()

        log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        log_img = self.logger.concatenate_images([log_img, input_img.cpu().numpy()[:, :, :, :]])
        self.logger.log_images(log_img, self.epoch, n_batch, num_batches, nrows=log_img.shape[0])
