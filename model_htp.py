import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from MolCLR.models.ginet_finetune import num_atom_type, num_chirality_tag

EPS = 1e-12

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class HTPFineTune(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        #dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        #log_dir = os.path.join('finetune', dir_name)
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]
        loss = self.criterion(pred, data.y.flatten())
        return loss

    def train_kfold(self, k_folds):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        dataset = self.dataset.get_dataset()

        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset.labels)):
            print(f"Fold {fold + 1}")
            print("-------------")

            train_loader = self.dataset.get_data_loader_with_indices(train_idx)
            test_loader = self.dataset.get_data_loader_with_indices(test_idx)        

            if self.config['model_type'] == 'gin':
                from MolCLR.models.ginet_finetune import GINet
                model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)
            elif self.config['model_type'] == 'gcn':
                from MolCLR.models.gcn_finetune import GCN
                model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
                model = self._load_pre_trained_weights(model)

            layer_list = []
            for name, param in model.named_parameters():
                if 'pred_head' in name:
                    #print(name, param.requires_grad)
                    layer_list.append(name)

            params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
            base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

            optimizer = torch.optim.Adam(
                [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
                self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
            )

            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

            # save config file
            _save_config_file(model_checkpoints_folder)

            n_iter = 0

            for epoch_counter in range(self.config['epochs']):
                for bn, data in enumerate(train_loader):
                    optimizer.zero_grad()

                    data = data.to(self.device)
                    loss = self._step(model, data, n_iter)                      

                    loss.backward()

                    optimizer.step()
                    n_iter += 1

                print(epoch_counter, loss.item())

            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f'model_{fold+1}.pth'))
            self.test_kfold(model, test_loader, model_fn=f'model_{fold+1}.pth')

    def test_kfold(self, model, test_loader, model_fn=None):
        if model_fn is None:
            model_fn = 'model.pth'
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', model_fn)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels)
        self.roc_auc = roc_auc_score(labels, predictions[:,1])
        print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, predictions[:,1])
        print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
        return valid_loss, roc_auc

    def htp_pred(self, model, model_path, test_loader, return_grad=False):
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions, grads = [], []
        for _, data in enumerate(test_loader):
            model.eval()
            if return_grad:   
                data = data.to(self.device)
                x, pred = self.run_model(model, data)
                pred = F.softmax(pred, dim=-1)
              
                pred_val = pred[:,1]
                pred_val.backward(retain_graph=True)

                grad = x.grad
                grad_abs = torch.sum(torch.abs(grad), -1).squeeze(0)
                grad_scale = grad_abs / (EPS + torch.norm(grad_abs, dim=-1))
                #grad_scale = 1 / (1 + torch.exp(-10 * (grad_scale - 0.5)))
                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    grads.append(grad_scale.detach().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    grads.append(grad_scale.cpu().detach().numpy())
            
            else:
                with torch.no_grad():
                    data = data.to(self.device)
                    _, pred = model(data)
                    pred = F.softmax(pred, dim=-1)

                    if self.device == 'cpu':
                        predictions.extend(pred.detach().numpy())
                    else:
                        predictions.extend(pred.cpu().detach().numpy())

        predictions = np.array(predictions)
        if return_grad:
            return predictions[:,1], grads
        else:
            return predictions[:,1], None
        
    def run_model(self, model, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        x1 = F.one_hot(x[:,0], num_classes=num_atom_type).float()
        x2 = F.one_hot(x[:,1], num_classes=num_chirality_tag).float()
        x = torch.cat([x1, x2], dim=-1)
        x.requires_grad = True
        h = x[:,:num_atom_type] @ model.x_embedding1.weight + \
            x[:,num_atom_type:] @ model.x_embedding2.weight

        for layer in range(model.num_layer):
            h = model.gnns[layer](h, edge_index, edge_attr)
            h = model.batch_norms[layer](h)
            if layer == model.num_layer - 1:
                h = F.dropout(h, model.drop_ratio, training=False)
            else:
                h = F.dropout(F.relu(h), model.drop_ratio, training=False)

        h = model.pool(h, data.batch)
        h = model.feat_lin(h)
        
        return x, model.pred_head(h)
