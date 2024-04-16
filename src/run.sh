# Start with trial prediction
# Either only visual, only audio, only ar*2, or ar vs visual+audio, or everything
WINDOW_OPTIONS=(90 90 150 150)
STEP_OPTIONS1=(60 90 90 150)
STEP_OPTIONS2=(30 90 30 150)

BY_USER_WINDOW_OPTIONS=(60 60 90 90 150 150)
BY_USER_STEP_OPTIONS=(30 60 30 60 30 90)

BY_SAMPLE_WINDOW_OPTIONS=(60 90 120 150)
BY_SAMPLE_STEP_OPTIONS=(60 90 120 150)


# # val only 
# for trial in "'[3,5]'" "'[2,4]'" "'[1,2]'" "'[5,6]'" "'[1,6]'" "'[2,5]'" "'[1,2,3,4,5,6]'"; do
# # first do by user, window size can be smaller
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
#     done
#     # by time
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
#     done
#     # then by sample
#     for ((i=0;i<4;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
#     done
# done


# # Secondly distance prediction
# # for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
# for phase in "'[4,5,6]'" "'[4]'" "'[5]'"; do
#     for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
#         for ((i=0;i<6;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
#         done
#         # by time
#         for ((i=0;i<6;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
#         done
#         # then by sample
#         for ((i=0;i<4;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
#         done
#     done
# done

# # Lastly ADHD
# # for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
# for phase in "'[4,5,6]'" "'[4]'" "'[5]'"; do
#     for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
#         for ((i=0;i<6;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
#         done
#         # by time
#         for ((i=0;i<6;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
#         done
#         # then by sample
#         for ((i=0;i<4;i++)); do
#             CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
#         done
#     done
# done

# val + test

# for trial in "'[3,5]'" "'[2,4]'" "'[1,2]'" "'[5,6]'" "'[1,6]'" "'[2,5]'" "'[1,2,3,4,5,6]'"; do
# # first do by user, window size can be smaller
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
#     done
#     # by time
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
#     done
#     # then by sample
#     for ((i=0;i<4;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
#     done
# done


# Secondly distance prediction
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
        done
    done
done

# Lastly ADHD
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample
        done
    done
done


# for trial in "'[3,5]'" "'[2,4]'" "'[1,2]'" "'[5,6]'" "'[1,6]'" "'[2,5]'" "'[1,2,3,4,5,6]'"; do
# # first do by user, window size can be smaller
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
#     done
#     # by time
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
#     done
#     # then by sample
#     for ((i=0;i<4;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
#     done
# done


# Secondly distance prediction
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
        done
    done
done

# Lastly ADHD
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --val_ratio 0.2 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
        done
    done
done

# val + test

# for trial in "'[3,5]'" "'[2,4]'" "'[1,2]'" "'[5,6]'" "'[1,6]'" "'[2,5]'" "'[1,2,3,4,5,6]'"; do
# # first do by user, window size can be smaller
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
#     done
#     # by time
#     for ((i=0;i<6;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
#     done
#     # then by sample
#     for ((i=0;i<4;i++)); do
#         CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/trial --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task trial --distraction_used_phases '[4,5]' --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
#     done
# done


# Secondly distance prediction
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
        done
    done
done

# Lastly ADHD
# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'"; do
for phase in "'[4,5]'"; do
    for trial in "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --smooth_targets
        done
        # by time
        for ((i=0;i<6;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --test_pattern test --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --smooth_targets
        done
        # then by sample
        for ((i=0;i<4;i++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --val_ratio 0.1 --test_ratio 0.1 --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_SAMPLE_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_SAMPLE_STEP_OPTIONS[i]} --distraction_split by_sample --smooth_targets
        done
    done
done