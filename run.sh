#!/bin/bash

RUN_NUM=11
target_emo="comfort"
gpu_id=0
ae_num_epochs=10
ae_target_epoch=10
est_num_epochs=5
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-video_name_list-fbf.csv"
###############################################

run_name="c_a_gt"

python src/notify-slack.py --message "$run_name start!"

for i in $(seq 1 $RUN_NUM)
do

    echo "----- Start $run_name fold $i/$RUN_NUM -----"

    # label_path="/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/emo-au-gaze-hp(video1-25)-gt.csv"
    # only_positive_label_path="/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/emo-au-gaze-hp(video1-25)-onlypos-gt.csv"

    # python src/train-autoencoder.py \
    # --run_name $run_name \
    # --fold $i \
    # --gpu_id $gpu_id \
    # --num_epochs $ae_num_epochs \
    # --only_positive_label_path $only_positive_label_path \
    # --video_name_list_path $video_name_list_path \


    # python src/extract-negative.py \
    # --run_name $run_name \
    # --fold $i \
    # --gpu_id $gpu_id \
    # --target_epoch $ae_target_epoch \
    # --label_path $label_path \
    # --only_positive_label_path $only_positive_label_path \
    # --video_name_list_path $video_name_list_path \

    use_pseudo_label="False"
    label_path="/mnt/iot-qnap3/mochida/medical-care/InitialModelConstruction/data/labels/PIMD_A/comfort-ausign-labels-fbf.csv"

    python src/train-estimator.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --use_pseudo_label $use_pseudo_label \
    --ae_target_epoch $ae_target_epoch \
    --gpu_id $gpu_id \
    --num_epochs $est_num_epochs \
    --label_path $label_path \
    --video_name_list_path $video_name_list_path \

    for j in $(seq 1 $est_num_epochs)
    do

        python src/test-estimator.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --gpu_id $gpu_id \
        --target_epoch $j \
        --label_path $label_path \
        --video_name_list_path $video_name_list_path \

    done

done

for k in $(seq 1 $est_num_epochs)
do

    python src/calculate-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $k \

done

python src/notify-slack.py --message "$run_name finish!"