import torch
import numpy as np
import pandas as pd
import random
import os

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
def get_video_name_list(video_name_list_path, fold=0, phase='train'):
    video_name_list = pd.read_csv(video_name_list_path)
    
    if fold == 0:
        video_name_list = video_name_list.columns.values.tolist()
    else:
        if phase == 'train':
            video_name_list = video_name_list.columns[video_name_list.iloc[fold - 1] == 0].values.tolist()
        elif phase == 'test':
            video_name_list = video_name_list.columns[video_name_list.iloc[fold - 1] == 1].values.tolist()
        
    return video_name_list

def str2bool(v):
    return v.lower() in ('true')

def get_video_name_and_frame_id(img_path):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    video_name, frame_id = basename.split('_')
    frame_id = int(frame_id)
    return video_name, frame_id

def get_sequence_img_path(current_img_path, window_size=30, current_position='tail'):
    img_path_list = []
    root_dir = os.path.dirname(current_img_path)
    
    video_name, current_frame = get_video_name_and_frame_id(current_img_path)
    
    if current_position == 'tail':
        for i in range(0, window_size):
            past_frame = current_frame - (window_size - i - 1)
            if past_frame < 0:
                img_path_list.append('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/interim/black.jpg')
            else:
                img_path_list.append(os.path.join(root_dir, f'{video_name}_{str(past_frame).zfill(4)}.jpg'))
    
    elif current_position == 'head':
        for i in range(0, window_size):
            new_frame = current_frame + i
            img_path = os.path.join(root_dir, f'{video_name}_{str(new_frame).zfill(4)}.jpg')
            if os.path.exists(img_path) == False:
                img_path_list.append('/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/interim/black.jpg')
            else:
                img_path_list.append(img_path)
            
    return img_path_list

def convert_label_to_binary(labels, target_emo):
    '''
    Args:
        labels: tensor
        target_emo: str 'comfort' or 'discomfort'
    '''
    if target_emo == 'comfort':
        labels = torch.where(labels == 2, torch.tensor(0), labels)
    elif target_emo == 'discomfort':
        labels = torch.where(labels == 1, torch.tensor(0), labels)
        labels = torch.where(labels == 2, torch.tensor(1), labels)
        
    return labels