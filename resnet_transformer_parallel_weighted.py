from operator import index
from turtle import pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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
import torch.multiprocessing as mp

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

def cleanup():
    dist.destroy_process_group()

class CustomDataset(Dataset):
    def __init__(self, clinical_data, labels, image_files, image_dir, transform=None, target_transform=None):
        super().__init__()

        self.clinical_data = clinical_data
        self.labels = labels
        self.image_files = image_files
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) :
        clinical = self.clinical_data[idx,:]
        label = self.labels[idx]
        img_file_path = self.image_dir + str(self.image_files[idx,0])+'_'+str(self.image_files[idx,1])+'.pt'
        image = img_tensor_loader(img_file_path)

        
        return clinical, label, image

def MyCollate_fn(batch):
    clinicals = []
    labels = []
    images = []

    for item in batch:
        clinicals.append(item[0])
        labels.append(item[1])
        images.append(item[2])

    clinicals = torch.tensor(np.array(clinicals), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    return clinicals, labels, images


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
        # self.device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

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




    def forward(self, x_images, x_clinical, rank):

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
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(rank)

        # src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True)



        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        h_s = torch.zeros((n, e)).to(rank)

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
        # self.device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

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


    def forward(self, x_images, x_clinical, rank):

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
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(rank)


        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        h_s = torch.zeros((n, e)).to(rank)

        for i in range(n):
            temp_output = outputs_trans[:lens_unpacked[i], i, :]
            h_s[i, :] = torch.mean(temp_output, dim=0)

        h_s = self.fc2(h_s)

        # '''影像特征和临床特征合并'''
        # h_s_mul = torch.cat((x_clinical, h_s), 1)
        # outputs = self.fc1(h_s_mul)
        outputs = self.fc1(h_s)

        return outputs, h_s





def train_rnt(
    rank, 
    world_size, 
    hyper_params,
    model,  
    data_train, 
    labels_train, 
    data_image_filename_train, 
    data_val, 
    labels_val, 
    data_image_filename_val, 
    data_test, 
    labels_test, 
    data_image_filename_test,
    best_model_dir,
    fold
):
    
    
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank

    # hyper_params = None
    # data_train = None
    # labels_train = None
    # data_image_filename_train = None
    # data_val = None
    # labels_val = None
    # data_image_filename_val = None
    # data_test = None
    # labels_test = None
    # data_image_filename_test = None


    model = model.to(rank)



    print('model load to gpu {}'.format(rank))
    ddp_model = DDP(model, device_ids=[rank])
    print('ddp_model load to gpu {}'.format(rank))
    dist.barrier()
    best_model_wts = copy.deepcopy(model.state_dict())
    lr = hyper_params['learning_rate']
    batch_size = hyper_params['batch_size']
    num_epochs = hyper_params['num_epochs']
    
    criterian = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    # optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=lr, betas=[0.9, 0.999])

    # optimizer = torch.optim.Adam([{'params':ddp_model.parameters()}, {'params':ddp_model.module.fc2.weight, 'weight_decay':0.5}], lr=lr, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5)

    model.train()
    total_loss = 0
    best_val_loss = float("inf")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")


    train_dataset = CustomDataset(clinical_data=data_train.values, labels=labels_train, image_files=data_image_filename_train.values,image_dir=config.local_CT_data_folder_path)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=MyCollate_fn, sampler=train_sampler)


    for epoch in range(num_epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # train_sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()

        for data in train_dataloader:
            # a = 1
            batch_start_time = time.time()

            clinicals, labels, images = data
            clinicals = clinicals.to(rank)
            labels = labels.to(rank)
            images_3 = []
            
     
            for img in images:
                img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(rank)
                images_3.append(img)



            optimizer.zero_grad()
            outputs, h_s = ddp_model(images_3, clinicals, rank)

            loss = criterian(outputs, labels) + 0.01 * torch.norm(ddp_model.module.fc1.weight)
            loss.backward()
            optimizer.step()
    
            probas = softmax(outputs)

            total_loss += loss.item()

            elapsed = time.time() - batch_start_time

            # print('Epoch {} rank {} total loss {}'.format(epoch, rank, total_loss))

        
        scheduler.step()
  
        
        if rank == 0:
            # train_loss, train_auc, train_ap = evaluate_rnt(model, criterian, data_train, labels_train, data_image_filename_train)
            val_loss, val_auc, val_ap, _, _ = evaluate_rnt(rank,ddp_model, criterian, data_val, labels_val, data_image_filename_val)
            test_loss, test_auc, test_ap, _, _ = evaluate_rnt(rank, ddp_model, criterian, data_test, labels_test, data_image_filename_test)
            print('Current rank {} learning rate: {}'.format(rank, optimizer.param_groups[0]['lr']))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
                    'valid AUC {:.3f}  | valid AP {:.3f} | test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(epoch, (time.time() - epoch_start_time), total_loss, val_loss, val_auc, val_ap, test_loss, test_auc, test_ap))
            print('-' * 89)


            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())

                path = best_model_dir + 'best_val_model_' + str(fold) + '.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                
                torch.save({
                    'model_state_dict': best_model_wts,
                }, path)

        

        
        total_loss = 0
        # print(f"Rank:{rank} waiting before the barrier")
        dist.barrier()
        # print(f"Rank:{rank} left the barrier")

    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    #     # # train_loss, train_auc, train_ap = evaluate_rnt(model, criterian, data_train, labels_train, data_image_filename_train)
    #     # val_loss, val_auc, val_ap, _, _ = evaluate_rnt(model, criterian, data_val, labels_val, data_image_filename_val)
    #     # test_loss, test_auc, test_ap, _, _ = evaluate_rnt(model, criterian, data_test, labels_test, data_image_filename_test)

        
        # if val_auc > best_val_auc:
        #     # best_model_dir = './model/clinical/rnt/'
        #     best_val_auc = val_auc
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     # path = best_model_dir + 'best_val_model.pt'
        #     # if not os.path.exists(best_model_dir):
        #     #     os.makedirs(best_model_dir)
        #     #
        #     # torch.save({
        #     #     'epoch': epoch,
        #     #     'model_state_dict': best_model_wts,
        #     #     'optimizer_state_dict': optimizer.state_dict(),
        #     #     'loss': val_loss,
        #     #     'lr': lr,
        #     #     'batch_size': batch_size,
        #     #
        #     # }, path)




    # # # load best model weights
    # # # model.load_state_dict(best_model_wts)
    
    cleanup()
    return None


def evaluate_rnt(rank, eval_model: nn.Module, criterian, data_val, labels_val, data_image_filename_val):
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    # eval_model.to(rank)
    eval_model.eval()
    data_image_file_path = config.local_CT_data_folder_path

    train_dataset = CustomDataset(clinical_data=data_val.values, labels=labels_val, image_files=data_image_filename_val.values,image_dir=config.local_CT_data_folder_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=MyCollate_fn)

    y_probs = np.array([])
    y_true = np.array([])
    total_loss = 0.0

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    for data in train_dataloader:
        clinicals, labels, images = data
        clinicals = clinicals.to(rank)
        labels = labels.to(rank)
        images_3 = []
        
    
        for img in images:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(rank)
            images_3.append(img)
        
        with torch.no_grad():

            outputs, _ = eval_model.module(images_3, clinicals,rank)
            # outputs= outputs.view(-1)
            loss = criterian(outputs, labels)
            # probas = sigmoid(outputs)
            probas = softmax(outputs)
            
            # if np.any(np.isnan(probas.view(-1).detach().cpu().numpy())):
            #     a=1
            # auc = roc_auc_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
            # ap = average_precision_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
    
        probas = probas.cpu().numpy()[:, 1]
        labels = labels.cpu().numpy()
        total_loss += loss.cpu().numpy()
        y_probs = np.append(y_probs, probas)
        y_true = np.append(y_true, labels)
        
    auc = roc_auc_score(y_true=y_true, y_score=y_probs)
    ap = average_precision_score(y_true=y_true, y_score=y_probs)

    return total_loss, auc, ap, y_probs, y_true



def test_rnt(model: nn.Module, criterian, data_test, labels_test, data_image_filename_test):
    
    device = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    data_image_file_path = config.local_CT_data_folder_path

    train_dataset = CustomDataset(clinical_data=data_test.values, labels=labels_test, image_files=data_image_filename_test.values,image_dir=config.local_CT_data_folder_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=MyCollate_fn)

    y_probs = np.array([])
    y_true = np.array([])
    total_loss = 0.0

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    for data in train_dataloader:
        clinicals, labels, images = data
        clinicals = clinicals.to(device)
        labels = labels.to(device)
        images_3 = []
        
    
        for img in images:
            img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            images_3.append(img)
        
        with torch.no_grad():

            outputs, _ = model(images_3, clinicals,device)
            # outputs= outputs.view(-1)
            loss = criterian(outputs, labels)
            # probas = sigmoid(outputs)
            probas = softmax(outputs)
            
            # if np.any(np.isnan(probas.view(-1).detach().cpu().numpy())):
            #     a=1
            # auc = roc_auc_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
            # ap = average_precision_score(y_true=y_true, y_score=probas.view(-1).detach().cpu().numpy()[:,1])
    
        probas = probas.cpu().numpy()[:, 1]
        labels = labels.cpu().numpy()
        total_loss += loss.cpu().numpy()
        y_probs = np.append(y_probs, probas)
        y_true = np.append(y_true, labels)
        
    auc = roc_auc_score(y_true=y_true, y_score=y_probs)
    ap = average_precision_score(y_true=y_true, y_score=y_probs)

    return total_loss, auc, ap, y_probs, y_true
