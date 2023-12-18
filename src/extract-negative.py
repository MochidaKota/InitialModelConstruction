import os
import pandas as pd
import argparse

import torch
import torch.nn as nn
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
    dataset = FeatList(
        labels_path=config.label_path,
        video_name_list=get_video_name_list(config.video_name_list_path, config.fold, 'train'),
        feats_path=config.feat_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    
    # define model (load trained model)
    input_dim = pd.read_pickle(config.feat_path).shape[1] - 1
    
    model = AutoEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim
    ).to(device)
    
    model_path = config.model_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}' + '/autoencoder.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print()
    print(f'--- Start scoring for fold{config.fold}-epoch{config.target_epoch}. ---')
    print()
    
    mse_list = []
    emotion_list = []
    img_path_list = []
    mse_loss = nn.MSELoss()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, img_paths, emotions = batch
            
            inputs = inputs.to(device)
            img_path_list += img_paths
            emotion_list += emotions.tolist()
            
            outputs = model(inputs)
            
            mse = mse_loss(inputs, outputs)
            mse_list.append(mse.item())
    
    print()      
    print(f'--- Finish scoring for fold{config.fold}-epoch{config.target_epoch}. ---')
    print()
              
    if config.other_run_name is None:
        repo_path_dir = config.repo_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    else:
        repo_path_dir = config.repo_path_prefix + config.other_run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    os.makedirs(repo_path_dir, exist_ok=True)
    
    # save mse scores
    scores_df = pd.DataFrame({'img_path':img_path_list, 'emotion':emotion_list, 'mse':mse_list})
    onlypositive_label_df = pd.read_csv(config.only_positive_label_path)
    onlypositive_label_df = onlypositive_label_df[onlypositive_label_df['video_name'].isin(get_video_name_list(config.video_name_list_path, config.fold, 'train'))]
    onlypositive_label_df = onlypositive_label_df[['img_path', 'emotion']]
    
    scores_df = scores_df[~scores_df['img_path'].isin(onlypositive_label_df['img_path'])]
    scores_df.to_csv(repo_path_dir + '/mse_scores.csv', index=False)
    
    # extract negative samples
    # negative samples are sampled from the bottom of mse scores
    # the number of negative samples is the same as the number of positive samples
    scores_df = scores_df.sort_values('mse', ascending=False)
    negative_df = scores_df.iloc[:onlypositive_label_df.shape[0]].copy()
    negative_df.loc[:, 'emotion'] = 0
    negative_df = negative_df[['img_path', 'emotion']]
    
    pseudo_label_df = pd.concat([onlypositive_label_df, negative_df])
    pseudo_label_df = pseudo_label_df.sort_values('img_path')
    pseudo_label_df = pseudo_label_df.reset_index(drop=True)
    pseudo_label_df.to_csv(repo_path_dir + '/pseudo_label.csv', index=False)
    
    print()
    print(f'--- Save mse scores and Extract negative samples for fold{config.fold}-epoch{config.target_epoch}. ---')
    print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default', help='run name')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='hidden dimension')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension')
    
    # test config
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--other_run_name', type=str, default=None, help='other run name')
    parser.add_argument('--target_epoch', type=int, default=0, help='target epoch')
    
    # path config
    parser.add_argument('--label_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/emo-au-gaze-hp(video1-25)-gt.csv', help='path to label.csv')
    parser.add_argument('--only_positive_label_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/emo-au-gaze-hp(video1-25)-onlypos-gt.csv', help='path to only_positive_label.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-video_name_list-fbf.csv', help='path to video_name_list.csv')
    parser.add_argument('--feat_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features.pkl')
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/snapshots/PIMD_A/', help='path to save model')
    parser.add_argument('--repo_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/report/PIMD_A/', help='path to save repo')
    
    config = parser.parse_args()
    
    main(config)