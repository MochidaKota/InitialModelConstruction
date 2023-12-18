import os
import pandas as pd
import yaml
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util import torch_fix_seed, get_video_name_list, str2bool, convert_label_to_binary
from network import MLPClassifier
from dataset import FeatList

def main(config):
    # fix random seed
    torch_fix_seed()
    
    # define device
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # define dataset and dataloader
    if config.use_pseudo_label == True:
        label_path = config.repo_path_prefix + config.run_name + f'/epoch{config.ae_target_epoch}' + f'/fold{config.fold}' + '/pseudo_label.csv'
    else:
        label_path = config.label_path
    
    train_dataset = FeatList(
        labels_path=label_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        feats_path=config.feat_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    if config.check_test_loss == True:
        test_dataset = FeatList(
            labels_path=config.label_path,
            video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'test'),
            feats_path=config.feat_path
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
        
    # define model
    input_dim = pd.read_pickle(config.feat_path).shape[1] - 1
    
    if config.hidden_dims is None:
        hidden_dims = [2048, 512]
    else:
        hidden_dims = config.hidden_dims
        
    model = MLPClassifier(
        num_classes=config.num_classes,
        input_dim=input_dim,
        hidden_dims=hidden_dims
    ).to(device)
    
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    
    # define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    print()
    print(f'--- Start training Estimator for fold{config.fold}. ---')
    print()
    
    if config.check_test_loss == True:
        history = {'epoch':[], 'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
        phases = ['train', 'test']
    else:
        history = {'epoch':[], 'train_loss':[], 'train_acc':[]}
        phases = ['train']
        
    for epoch in range(config.num_epochs):
            
        history['epoch'].append(epoch+1)
        
        # define directory to save model
        save_model_dir = config.model_path_prefix + config.run_name + f'/epoch{epoch+1}' + f'/fold{config.fold}'
        os.makedirs(save_model_dir, exist_ok=True)
        
        # start training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader
                
            running_loss = 0.0
            running_corrects = 0
            start = time.time()
            
            for i, batch in enumerate(dataloader):
                # get batch
                feats, _, emos = batch
                feats = feats.to(device)
                emos = convert_label_to_binary(emos, config.target_emo).float().to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(feats)
                    outputs = outputs.view(-1)
                    loss = criterion(outputs, emos)
                    preds = torch.round(torch.sigmoid(outputs))
                    
                    # backward
                    if phase == 'train':
                        loss.backward()   
                        optimizer.step()
                
                # sum iteration loss and corrects       
                running_loss += loss.item() * emos.size(0)
                running_corrects += torch.sum(preds == emos.data).double().item()
                
                # release GPU memory
                del feats, emos, outputs, preds
                torch.cuda.empty_cache()
            
            # calculate epoch loss and accuracy
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            
            # store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            elif phase == 'test':
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)
                
            # display epoch loss and accuracy
            print(f'epoch: {epoch+1}, phase: {phase}, loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, time: {time.time()-start:.1f}[sec]')
            
            # save model
            if phase == 'train':
                torch.save(model.state_dict(), save_model_dir + '/emotion_estimator.pth')
                    
    print()
    print(f'--- Finish training Estimator for fold{config.fold}. ---')
    print()
    
    # save history
    history = pd.DataFrame(history)
    save_history_dir = config.repo_path_prefix + config.run_name + f'/history/fold{config.fold}'
    os.makedirs(save_history_dir, exist_ok=True)
    history.to_csv(save_history_dir + '/history_est.csv')
    
    # save config
    save_repo_dir = config.repo_path_prefix + config.run_name
    os.makedirs(save_repo_dir, exist_ok=True)
    with open(save_repo_dir + '/config_est.yaml', 'w') as f:
        yaml.dump(config.__dict__, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default', help='run name')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--use_pseudo_label', type=str2bool, default=True, help='whether to use pseudo label or not')
    parser.add_argument('--ae_target_epoch', type=int, default=10, help='use autoencoder trained at this epoch')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--hidden_dims', nargs='*', type=int, default=None, help='hidden dimensions')
    # training config
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--check_test_loss', type=str2bool, default=True, help='whether to check test loss or not')
    
    # path config
    parser.add_argument('--label_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-labels-fbf.csv', help='path to label.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-video_name_list-fbf.csv', help='path to video_name_list.csv')
    parser.add_argument('--feat_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features.pkl')
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/snapshots/PIMD_A/', help='path to save model')
    parser.add_argument('--repo_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/report/PIMD_A/', help='path to save repo')
    
    config = parser.parse_args()
    
    main(config)