from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_MultiLabeling(Exp_Basic):
    def __init__(self, args):
        super(Exp_MultiLabeling, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.enc_in
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # ref. https://qiita.com/koshian2/items/ab5e0c68a257585d7c6f
        # 多ラベル分類の場合は、BCELossを使う
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def mean_squared_error(self, preds, trues):
        return np.mean(np.square(preds - trues))

    def mean_squared_error_with_threshold(self, preds, trues, threshold=0.5):
        preds = preds > threshold
        return np.mean(np.square(preds - trues))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.float().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.sigmoid(preds)
        trues = trues.cpu().numpy()
        err = self.mean_squared_error(probs.cpu().numpy(), trues)
        err2 = self.mean_squared_error_with_threshold(probs.cpu().numpy(), trues)

        self.model.train()
        return total_loss, err, err2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask)
                loss = criterion(outputs, label.float().to(self.device))
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_err, val_err2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_err, test_err2 = self.vali(test_data, test_loader, criterion)

            print("""
Epoch: {0}, Steps: {1}
Train
    Loss: {2:.3f}
Validation
    Loss: {3:.3f}
    MSE: {4:.3f}
    MSE2: {5:.3f}
Test
    Loss: {6:.3f}
    MSE: {7:.3f}
    MSE2: {8:.3f}
""".format(epoch + 1, train_steps, train_loss, vali_loss, val_err, val_err2, test_loss, test_err, test_err2))
            early_stopping(val_err, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.sigmoid(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        trues = trues.cpu().numpy()
        err = self.mean_squared_error(probs.cpu().numpy(), trues)
        err2 = self.mean_squared_error_with_threshold(probs.cpu().numpy(), trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('MSE: {}'.format(err))
        print('MSE2: {}'.format(err2))
        file_name = 'result_multi_labeling.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('MSE: {}'.format(err))
        f.write('MSE2: {}'.format(err2))
        f.write('\n')
        f.write('\n')
        f.close()
        return
