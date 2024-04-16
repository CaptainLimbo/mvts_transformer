# Start with trial prediction
# Either only visual, only audio, only ar*2, or ar vs visual+audio, or everything
BY_USER_WINDOW_OPTIONS=(60 120)
BY_USER_STEP_OPTIONS=(30 30)
RANDOM_SEEDS=(2024 2025 2026 2027 2028)



for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'" "'[6]'"; do
    for trial in  "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((j=0;j<5;j++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size 120 --distraction_step_size 30 --distraction_split by_user --seed ${RANDOM_SEEDS[j]} --smooth_targets
        done
        # by time
        for ((i=0;i<2;i++)); do
            for ((j=0;j<5;j++)); do
                CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --seed ${RANDOM_SEEDS[j]} --smooth_targets
            done
        done
    done
done


for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'" "'[6]'"; do
    for trial in  "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
        for ((j=0;j<5;j++)); do
            CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size 120 --distraction_step_size 30 --distraction_split by_user --seed ${RANDOM_SEEDS[j]} --smooth_targets
        done
        # by time
        for ((i=0;i<2;i++)); do
            for ((j=0;j<5;j++)); do
                CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --seed ${RANDOM_SEEDS[j]} --smooth_targets
            done
        done
    done
done



# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'" "'[6]'"; do
#     for trial in  "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
#         for ((i=0;i<3;i++)); do
#             for ((j=0;j<5;j++)); do
#                 CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --seed ${RANDOM_SEEDS[j]} 
#             done
#         done
#         # by time
#         for ((i=0;i<3;i++)); do
#             for ((j=0;j<5;j++)); do
#                 CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/distance --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task distance --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --seed ${RANDOM_SEEDS[j]}
#             done
#         done
#     done
# done


# for phase in "'[4,5]'" "'[4,5,6]'" "'[4]'" "'[5]'" "'[6]'"; do
#     for trial in  "'[2,3,4,5]'" "'[1,2,3,4,5,6]'"; do
#         for ((i=0;i<3;i++)); do
#             for ((j=0;j<5;j++)); do
#                 CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_user --seed ${RANDOM_SEEDS[j]}
#             done
#         done
#         # by time
#         for ((i=0;i<3;i++)); do
#             for ((j=0;j<5;j++)); do
#                 CUDA_VISIBLE_DEVICES=2 python src/main.py --output_dir experiments --comment "classification from Scratch" --name trial_fromScratch --records_file Classification_records.xls --data_class ar --data_dir datasets/Distraction/adhd --pattern train --val_pattern val --epochs 100 --lr 5e-4 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric accuracy --distraction_task adhd --distraction_used_phases $phase --distraction_used_trials $trial --distraction_window_size ${BY_USER_WINDOW_OPTIONS[i]} --distraction_step_size ${BY_USER_STEP_OPTIONS[i]} --distraction_split by_time --seed ${RANDOM_SEEDS[j]}
#             done
#         done
#     done
# done
