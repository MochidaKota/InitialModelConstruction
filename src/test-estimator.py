import os
import pandas as pd
import yaml
import argparse
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

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
    
    # define model (load trained model)
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
    
    model_path = config.model_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}' + '/emotion_estimator.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print()
    print(f'--- Start test Estimator for fold{config.fold}-epoch{config.target_epoch}. ---')
    print()
    
    emo_temp_list = []
    emo_gt_list = []
    emo_pred_list= []
    emo_posterior_list = []
    img_path_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get batch
            feats, img_paths, emos = batch
            feats = feats.to(device)
            img_path_list += img_paths
            emo_temp_list += emos.tolist()
            emos = convert_label_to_binary(emos, config.target_emo).to(device)
            
            # forward
            outputs = model(feats)
            outputs = outputs.view(-1)
            outputs = torch.sigmoid(outputs)
            preds = torch.round(outputs)
            
            # store outputs
            emo_gt_list += emos.detach().cpu().numpy().tolist()
            emo_pred_list += preds.detach().cpu().numpy().tolist()
            emo_posterior_list += outputs.detach().cpu().numpy().tolist()
            
            # release memory
            del feats, emos, outputs, preds
            torch.cuda.empty_cache()
            
    print()
    print(f'--- Finish test Estimator for fold{config.fold}-epoch{config.target_epoch}. ---')
    print()
    
    # calculate metrics
    if config.other_run_name is None:
        repo_path_dir = config.repo_path_prefix + config.run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    else:
        repo_path_dir = config.repo_path_prefix + config.other_run_name + f'/epoch{config.target_epoch}' + f'/fold{config.fold}'
    os.makedirs(repo_path_dir, exist_ok=True)
    
    precision = precision_score(emo_gt_list, emo_pred_list)
    recall = recall_score(emo_gt_list, emo_pred_list)
    f1 = f1_score(emo_gt_list, emo_pred_list)
    accuracy = accuracy_score(emo_gt_list, emo_pred_list)
    roc_auc = roc_auc_score(emo_gt_list, emo_posterior_list)
    pre, rec, _ = precision_recall_curve(emo_gt_list, emo_posterior_list)
    pr_auc = auc(rec, pre)
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')
    print(f'f1: {f1:.4f}')
    print(f'accuracy: {accuracy:.4f}')
    print(f'roc_auc: {roc_auc:.4f}')
    print(f'pr_auc: {pr_auc:.4f}')
    
    clf_report_df = pd.DataFrame([[precision, recall, f1, accuracy, roc_auc, pr_auc]], columns=["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"])
    clf_report_df.to_csv(repo_path_dir + '/clf_report.csv', index=False)
    
    cm = confusion_matrix(emo_gt_list, emo_pred_list)
    cm_df = pd.DataFrame(cm, index=['negative', 'positive'], columns=['negative', 'positive'])
    cm_df.to_csv(repo_path_dir + '/confusion_matrix.csv', index=False)
    
    pred_list = []
    for i in range(len(emo_pred_list)):
        pred_list.append([emo_temp_list[i]] + [emo_pred_list[i]] + [emo_posterior_list[i]] + [img_path_list[i]])
    pred_df = pd.DataFrame(pred_list, columns=["emo_gt","emo_pred", "emo_pos", "img_path"])
    pred_df.to_csv(repo_path_dir + "/" + f"pred.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model config
    parser.add_argument('--run_name', type=str, default='default', help='run name')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--hidden_dims', nargs='*', type=int, default=None, help='hidden dimensions')
    
    # test config
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--other_run_name', type=str, default=None, help='other run name')
    parser.add_argument('--target_epoch', type=int, default=0, help='target epoch')
    
    # path config
    parser.add_argument('--label_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-labels-fbf.csv', help='path to label.csv')
    parser.add_argument('--video_name_list_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-video_name_list-fbf.csv', help='path to video_name_list.csv')
    parser.add_argument('--feat_path', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/processed/PIMD_A/JAANet_feature.pkl', help='path to features.pkl')
    parser.add_argument('--model_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/snapshots/PIMD_A/', help='path to save model')
    parser.add_argument('--repo_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/report/PIMD_A/', help='path to save repo')
    
    config = parser.parse_args()
    
    main(config)