import os
import pandas as pd
import yaml
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util import torch_fix_seed, get_video_name_list, str2bool
from network import AutoEncoder
from dataset import FeatList

def main(config):
    # fix random seed
    torch_fix_seed()
    
    # define device
    device = torch.device('cuda:{}'.format(config.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # define dataset and dataloader
    train_dataset = FeatList(
        labels_path=config.only_positive_label_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        feats_path=config.feat_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    
    if config.check_test_loss == True:
        test_dataset = FeatList(
            labels_path=config.only_positive_label_path,
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
    
    model = AutoEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim
    ).to(device)
    
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    
    # define loss function
    criterion = nn.MSELoss()
    
    print()
    print(f'--- Start training AutoEncoder for fold{config.fold}. ---')
    print()
    
    if config.check_test_loss == True:
        history = {'epoch':[], 'train_loss':[], 'test_loss':[]}
        phases = ['train', 'test']
    else:    
        history = {'epoch':[], 'train_loss':[]}
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
            start = time.time()
            
            for i, batch in enumerate(dataloader):
                # get batch data
                inputs, _, _ = batch
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    
                    # backward
                    if phase == 'train':
                        loss.backward()   
                        optimizer.step()
                
                # sum iteration loss       
                running_loss += loss.item() * inputs.size(0)
            
            # calculate epoch loss   
            epoch_loss = running_loss / len(dataloader.dataset)
            
            # store epoch loss
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            elif phase == 'test':
                history['test_loss'].append(epoch_loss)
            
            # print epoch loss   
            print(f"epoch:{epoch+1}/{config.num_epochs} phase:{phase} loss:{epoch_loss:.4f} time:{time.time()-start:.1f}[sec]")
        
            # save model
            if phase == 'train':
                torch.save(model.state_dict(), save_model_dir + '/autoencoder.pth')
    
    print()          
    print(f'--- Finished training AutoEncoder for fold{config.fold}. ---')
    print()
    
    # save history
    history = pd.DataFrame(history)
    save_history_dir = config.repo_path_prefix + config.run_name + f'/history/fold{config.fold}'
    os.makedirs(save_history_dir, exist_ok=True)
    history.to_csv(save_history_dir + '/history_ae.csv')
    
    # save config as yaml
    save_repo_dir = config.repo_path_prefix + config.run_name
    os.makedirs(save_repo_dir, exist_ok=True)
    with open(save_repo_dir + '/config_ae.yaml', 'w') as f:
        yaml.dump(config.__dict__, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default', help='run name')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='hidden dimension')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension')
    
    # training config
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--check_test_loss', type=str2bool, default=True, help='whether to check test loss or not')
    
    # path config
    parser.add_argument('--only_positive_label_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/emo-au-gaze-hp(video1-25)-onlypos-gt.csv', help='path to only_positive_label.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-video_name_list-fbf.csv', help='path to video_name_list.csv')
    parser.add_argument('--feat_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features.pkl')
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/snapshots/PIMD_A/', help='path to save model')
    parser.add_argument('--repo_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/report/PIMD_A/', help='path to save repo')
    
    config = parser.parse_args()
    
    main(config)