import glob
import os
import pandas as pd
import argparse
from util import str2bool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def main(config):
    repo_path_dir = config.repo_path_prefix + config.run_name + '/' + f'epoch{config.target_epoch}'
    os.makedirs(repo_path_dir, exist_ok=True)
    report = pd.DataFrame()
    pred = pd.DataFrame()
    
    for i in range(len(glob.glob(repo_path_dir + "/fold*"))):
        fold_dir = repo_path_dir + "/fold{}".format(i + 1)
        fold_report = pd.read_csv(fold_dir + f"/clf_report.csv")
        fold_pred = pd.read_csv(fold_dir + f"/pred.csv")
        report = pd.concat([report, fold_report], axis=0)
        pred = pd.concat([pred, fold_pred], axis=0)
    
    report.index = ["fold{}".format(i + 1) for i in range(report.shape[0])]
    report_mean = pd.DataFrame(report.mean()).T
    report_mean.index = ["mean"]
    report = pd.concat([report, report_mean], axis=0)
    report.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    
    pred = pred.sort_values('img_path')
    pred = pred.reset_index(drop=True)
    gt_list = []
    if config.target_emo == "comfort":
        # copy pred["emo_gt"] to gt_list
        gt_list = pred["emo_gt"].copy()
        # replace 2 to 0
        gt_list = gt_list.replace(2, 0)
    elif config.target_emo == "discomfort":
        # copy pred["emo_gt"] to gt_list
        gt_list = pred["emo_gt"].copy()
        # replace 1 to 0
        gt_list = gt_list.replace(1, 0)
        # replace 2 to 1
        gt_list = gt_list.replace(2, 1)
        
    # calc metrics gt_list and pred["emo_pred"]
    # roc_auc, pr_auc is calculated by pred["emo_pos"]
    acc = accuracy_score(gt_list, pred["emo_pred"])
    precision = precision_score(gt_list, pred["emo_pred"])
    recall = recall_score(gt_list, pred["emo_pred"])
    f1 = f1_score(gt_list, pred["emo_pred"])
    fpr, tpr, _ = roc_curve(gt_list, pred["emo_pos"])
    roc_auc = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(gt_list, pred["emo_pos"])
    pr_auc = auc(rec, pre)
    
    # save confusion matrix
    emo_cm = confusion_matrix(gt_list, pred["emo_pred"])
    if config.target_emo == 'comfort':
        label_list = ['not comfort', 'comfort']
    elif config.target_emo == 'discomfort':
        label_list = ['not discomfort', 'discomfort']
    emo_cm_df = pd.DataFrame(emo_cm, index=label_list, columns=label_list)
    sns.heatmap(emo_cm_df, annot=True, cmap='Reds', fmt='g', annot_kws={"size": 10}, vmin=0, vmax=len(pred))
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.tight_layout()
    plt.savefig(repo_path_dir + "/" + f"confusion_matrix.png")
    plt.close()
    
    # save roc curve
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%roc_auc)
    plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label='Random ROC curve (area = %.2f)'%0.5, linestyle = '--', color = 'gray')
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.tight_layout()
    plt.savefig(repo_path_dir + "/" + f"roc_curve.png")
    plt.close()
    
    # save pr curve
    plt.plot(rec, pre, label='PR curve (area = %.2f)'%pr_auc)
    plt.legend()
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.tight_layout()
    plt.savefig(repo_path_dir + "/" + f"pr_curve.png")
    plt.close()
    
    # save pred
    pred.to_csv(repo_path_dir + f"/pred_all.csv", index=False)
        
    # save metrics as a csv file
    metrics = pd.DataFrame([precision, recall, f1, acc, roc_auc, pr_auc]).T
    metrics.columns = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
    metrics.to_csv(repo_path_dir + f"/metrics.csv", index=False)
    
    # save report
    report.to_csv(repo_path_dir + f"/metrics(macro_ave).csv")
    
    print()
    print(f'--- calculate metrics for epoch{config.target_epoch} ---')
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str, default='default', help='run name')
    parser.add_argument('--target_emo', type=str, default='comfort', choices=['comfort', 'discomfort'], help='target emotion')
    parser.add_argument('--target_epoch', type=int, default=0, help='target epoch')
    parser.add_argument('--repo_path_prefix', type=str, default='/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/report/PIMD_A/', help='repo path prefix')
    
    config = parser.parse_args()
    
    main(config)