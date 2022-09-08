from operator import index
from turtle import pd
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import math
import time
import copy
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import config
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def set_parameter_requires_grad(model, num_frozen_layer):
    if num_frozen_layer < 0:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for i, param in enumerate(model.parameters()):
            if i < num_frozen_layer:
                param.requires_grad = False


def img_tensor_loader(path:str):
    return torch.load(path)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class resnet_transformer(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 hidden_feature_size=128,
                 dropout = 0.1,
                 resnet_type='50',
                 use_pretrained = True,
                 ):
        super(resnet_transformer, self).__init__()
        self.device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

        self.num_classes = num_classes
        self.hidden_feature_size = hidden_feature_size

        if resnet_type == '101':
            self.resnet = models.resnet101(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '50':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '34':
            self.resnet = models.resnet34(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '18':
            self.resnet = models.resnet18(weights=models.ResNet50_Weights.DEFAULT)
        else:
            Exception('No proper resnet type set!')
        set_parameter_requires_grad(self.resnet, num_frozen_layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=self.hidden_feature_size)

        self.pos_encoder = PositionalEncoding(d_model=self.hidden_feature_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.fc2 = nn.Linear(in_features=self.hidden_feature_size, out_features=8)
        self.fc1 = nn.Linear(in_features=8+35, out_features=self.num_classes)




    def forward(self, x_images, x_clinical, mask=None):

        deep_features = []
        for i, x_image in enumerate(x_images):
            # x_image = x_image[:mask[i,:],:,:,:]
            x_image = self.resnet.conv1(x_image)
            x_image = self.resnet.bn1(x_image)
            x_image = self.resnet.relu(x_image)
            x_image = self.resnet.maxpool(x_image)

            x_image = self.resnet.layer1(x_image)
            x_image = self.resnet.layer2(x_image)
            x_image = self.resnet.layer3(x_image)
            x_image = self.resnet.layer4(x_image)

            x_image = self.resnet.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet.fc(x_image)
            deep_features.append(x_image)

        '''
        为了测试经过pack_sequence之后LSTM的输出hn是否还是batch中原始顺序输出的
        事实证明是的，因此可直接在hn基础上进行预测分类，并能够和labels对应
        '''
        deep_features_pack = pack_sequence(sequences=deep_features, enforce_sorted=False)
        # outputs, (hn, cn) = self.lstm(deep_features_pack)
        outputs_unpacked, lens_unpacked = pad_packed_sequence(deep_features_pack, batch_first=False, padding_value=0.0)

        #
        # # '''添加标志位'''
        # # cls = torch.zeros((1, n, e)).to(device)
        # # cls[0, :, 0] = 1.0
        # # outputs_unpacked = torch.cat((cls, outputs_unpacked), 0)
        # # s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        # # src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        # # for i in range(lens_unpacked.shape[0]):
        # #     src_key_padding_mask[i, lens_unpacked[i]+1:] = True
        #
        s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(self.device)


        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        h_s = torch.zeros((n, e)).to(self.device)

        for i in range(n):
            temp_output = outputs_trans[:lens_unpacked[i], i, :]
            h_s[i, :] = torch.mean(temp_output, dim=0)

        h_s = self.fc2(h_s)

        '''影像特征和临床特征合并'''

        h_s_mul = torch.cat((x_clinical, h_s), 1)
        outputs = self.fc1(h_s_mul)


        return outputs, h_s

class resnet_transformer_imgOnly(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 hidden_feature_size=128,
                 dropout = 0.1,
                 resnet_type='50',
                 use_pretrained = True,
                 ):
        super(resnet_transformer_imgOnly, self).__init__()
        self.device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

        self.num_classes = num_classes
        self.hidden_feature_size = hidden_feature_size

        if resnet_type == '101':
            self.resnet = models.resnet101(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '50':
            # self.resnet = models.resnet50(pretrained=use_pretrained)
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '34':
            self.resnet = models.resnet34(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_type == '18':
            self.resnet = models.resnet18(weights=models.ResNet50_Weights.DEFAULT)
        else:
            Exception('No proper resnet type set!')
        set_parameter_requires_grad(self.resnet, num_frozen_layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=self.hidden_feature_size)

        self.pos_encoder = PositionalEncoding(d_model=self.hidden_feature_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.fc2 = nn.Linear(in_features=self.hidden_feature_size, out_features=8)
        self.fc1 = nn.Linear(in_features=8, out_features=self.num_classes)




    def forward(self, x_images, x_clinical):

        deep_features = []
        for x_image in x_images:
            x_image = self.resnet.conv1(x_image)
            x_image = self.resnet.bn1(x_image)
            x_image = self.resnet.relu(x_image)
            x_image = self.resnet.maxpool(x_image)

            x_image = self.resnet.layer1(x_image)
            x_image = self.resnet.layer2(x_image)
            x_image = self.resnet.layer3(x_image)
            x_image = self.resnet.layer4(x_image)

            x_image = self.resnet.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet.fc(x_image)
            deep_features.append(x_image)

        '''
        为了测试经过pack_sequence之后LSTM的输出hn是否还是batch中原始顺序输出的
        事实证明是的，因此可直接在hn基础上进行预测分类，并能够和labels对应
        '''
        deep_features_pack = pack_sequence(sequences=deep_features, enforce_sorted=False)
        # outputs, (hn, cn) = self.lstm(deep_features_pack)
        outputs_unpacked, lens_unpacked = pad_packed_sequence(deep_features_pack, batch_first=False, padding_value=0.0)

        #
        # # '''添加标志位'''
        # # cls = torch.zeros((1, n, e)).to(device)
        # # cls[0, :, 0] = 1.0
        # # outputs_unpacked = torch.cat((cls, outputs_unpacked), 0)
        # # s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        # # src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        # # for i in range(lens_unpacked.shape[0]):
        # #     src_key_padding_mask[i, lens_unpacked[i]+1:] = True
        #
        s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(self.device)


        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        h_s = torch.zeros((n, e)).to(self.device)

        for i in range(n):
            temp_output = outputs_trans[:lens_unpacked[i], i, :]
            h_s[i, :] = torch.mean(temp_output, dim=0)

        h_s = self.fc2(h_s)

        # '''影像特征和临床特征合并'''
        # h_s_mul = torch.cat((x_clinical, h_s), 1)
        # outputs = self.fc1(h_s_mul)
        outputs = self.fc1(h_s)

        return outputs, h_s


def train_rnt(hyper_params, model: nn.Module, data_train, labels_train, data_image_filename_train, data_val, labels_val, data_image_filename_val, data_test, labels_test, data_image_filename_test):
    
    
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    data_image_file_path = config.local_CT_data_folder_path

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    sample_size = data_train.shape[0]
    num_epochs = hyper_params['num_epochs']
    criterian = nn.CrossEntropyLoss()

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    best_model = None


    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        epoch_start_time = time.time()
        for i in range(sample_size//batch_size+1):
            batch_start_time = time.time()
            star_idx = i*batch_size
            end_idx = i*batch_size+batch_size if i*batch_size+batch_size<=sample_size+1 else sample_size+1
            data_batch = data_train.values[star_idx:end_idx, :]
            labels_batch = labels_train[star_idx:end_idx]
            data_batch = torch.tensor(data_batch, dtype=torch.float32).to(device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)
            
            data_image_filename_batch = data_image_filename_train.values[star_idx:end_idx, :]
            data_image_batch = []
     
            for i in range(data_image_filename_batch.shape[0]):
                img_path = data_image_file_path+str(data_image_filename_batch[i,0])+'_'+str(data_image_filename_batch[i,1])+'.pt'
                img = img_tensor_loader(img_path)
                img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                data_image_batch.append(img)



            optimizer.zero_grad()
            outputs, h_s = model(data_image_batch, data_batch)

            loss = criterian(outputs, labels_batch)
            loss.backward()
            optimizer.step()
    
            probas = softmax(outputs)

            total_loss += loss.item()

            elapsed = time.time() - batch_start_time


        scheduler.step()
        print(optimizer.param_groups[0]['lr'])
            # print('| epoch {:3d} | {:5d}/{:5d} batches | '
            #  'lr {:02.2f} | ms/batch {:5.2f} | '
            #  'mean_loss {:5.2f}'.format(
            #     epoch, i, sample_size//batch_size+1, scheduler.get_last_lr()[0],
            #                   elapsed * 1000, total_loss/(i+1)))


        # train_loss, train_auc, train_ap = evaluate_rnt(model, criterian, data_train, labels_train, data_image_filename_train)
        val_loss, val_auc, val_ap, _ = evaluate_rnt(model, criterian, data_val, labels_val, data_image_filename_val)
        test_loss, test_auc, test_ap, _ = evaluate_rnt(model, criterian, data_test, labels_test, data_image_filename_test)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
              'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(epoch, (time.time() - epoch_start_time), total_loss, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap))
        print('-' * 89)

        total_loss = 0

        if val_auc > best_val_auc:
            # best_model_dir = './model/clinical/rnt/'
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            # path = best_model_dir + 'best_val_model.pt'
            # if not os.path.exists(best_model_dir):
            #     os.makedirs(best_model_dir)
            #
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': best_model_wts,
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': val_loss,
            #     'lr': lr,
            #     'batch_size': batch_size,
            #
            # }, path)




    # load best model weights
    # model.load_state_dict(best_model_wts)


    return best_val_auc, best_val_ap, best_model_wts


def evaluate_rnt(eval_model: nn.Module, criterian, data_val, labels_val, data_image_filename_val):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    eval_model.to(device)
    eval_model.eval()
    data_image_file_path = config.local_CT_data_folder_path

    data_val = torch.tensor(data_val.values, dtype=torch.float32).to(device)
    labels_val = torch.tensor(labels_val, dtype=torch.long).to(device)

    data_image_filename_batch = data_image_filename_val.values
    data_image_batch = []
    image_slice_num = []
    for i in range(data_image_filename_batch.shape[0]):
        img_path = data_image_file_path + str(data_image_filename_batch[i, 0]) + '_' + str(
            data_image_filename_batch[i, 1]) + '.pt'
        img = img_tensor_loader(img_path)
        img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        data_image_batch.append(img)


    total_loss = 0.
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    y_true = labels_val.detach().cpu().numpy()
    with torch.no_grad():
        outputs, _ = eval_model(data_image_batch, data_val)
        # outputs= outputs.view(-1)
        total_loss = criterian(outputs, labels_val)
        # probas = sigmoid(outputs)
        probas = softmax(outputs)
        probas = probas.detach().cpu().numpy()[:, 1]
        # if np.any(np.isnan(probas.view(-1).detach().cpu().numpy())):
        #     a=1
        # auc = roc_auc_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
        # ap = average_precision_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
        auc = roc_auc_score(y_true=y_true, y_score=probas)
        ap = average_precision_score(y_true=y_true, y_score=probas)

    return total_loss, auc, ap, probas
