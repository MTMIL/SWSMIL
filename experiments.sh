#!/bin/bash

P_P="./code/SWS-MIL/main_mix.py"
LOGDIR="./code/SWS-MIL/logger/ablation/c16/20240521-draw_instance"
CKPT="./code/SWS-MIL/ckpt/draw_instance_20240521"

# python $P_P --experiment_name c16_num2_label2_0 --device_ids 0 --fold 0 --max_pseudo_num 2 --pseudo_label_num 2 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_num4_label4_1 --device_ids 0 --fold 1 --max_pseudo_num 4 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_num6_label4_1 --device_ids 0 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_num8_label4_1 --device_ids 0 --fold 1 --max_pseudo_num 8 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_num10_label4_1 --device_ids 0 --fold 1 --max_pseudo_num 10 --pseudo_label_num 4 --ts_dir $LOGDIR &

# ablation of bagnum
# python $P_P --experiment_name c16_blank_num6_label4_2 --ckpt_dir $CKPT --device_ids 1 --mean_teacher --ada_pse 0 --max_merge_num 0 --fold 2 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &

python $P_P --experiment_name c16_noise_num6_label4_0 --ckpt_dir $CKPT --device_ids 3 --fold 0 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
python $P_P --experiment_name c16_mix_num6_label4_0 --ckpt_dir $CKPT --device_ids 3 --fold 0 --Merge 0 --Mix 1 --Noise 0 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_mix_num6_label4_2 --device_ids 1 --fold 2 --Merge 0 --Mix 1 --Noise 0 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &

# python $P_P --experiment_name c16_mix_num6_label4_0 --device_ids 2 --fold 0 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_mix_num6_label4_1 --device_ids 2 --fold 1 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_mix_num6_label4_2 --device_ids 2 --fold 2 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &

# python $P_P --experiment_name c16_mix_num6_label4_1 --device_ids 0 --fold 1 --Merge 0 --Mix 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_mix_num6_label4_2 --device_ids 0 --fold 2 --Merge 0 --Mix 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &

# python $P_P --experiment_name c16_noise_num6_label4_0 --device_ids 1 --fold 0 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --ckpt_dir $CKPT --experiment_name c16_noise_num6_label4_1 --device_ids 4 --fold 1 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_noise_num6_label4_2 --device_ids 3 --fold 2 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --ckpt_dir $CKPT --experiment_name c16_noise_num6_label4_1 --device_ids 4 --fold 1 --Merge 1 --Mix 0 --Noise 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
#!/bin/bash

P_P="./code/SWS-MIL/code/MT-PMIL/main.py"
LOGDIR="./code/SWS-MIL/logger/ablation/c16/20240521-draw_instance"
CKPT="./code/SWS-MIL/ckpt/draw_instance_20240521"


# ablation of bagnum

# python $P_P --ckpt_dir $CKPT --experiment_name c16_num6_blank_label4_1 --mean_teacher --ada_ssl_th --max_merge_num 0 --device_ids 2 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_blank_num6_label4_1 --ckpt_dir $CKPT --device_ids 1 --mean_teacher --ada_ssl_th --max_merge_num 0 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_ada_num6_label4_1 --ckpt_dir $CKPT --device_ids 1 --mean_teacher --max_merge_num 0 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_all_num6_label4_1 --ckpt_dir $CKPT --device_ids 1 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &



# 20240520
python $P_P --experiment_name c16_blank_num6_label4_0 --ckpt_dir $CKPT --device_ids 0 --mean_teacher --ada_pse 0 --max_merge_num 0 --fold 0 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
python $P_P --experiment_name c16_ada_num6_label4_0 --ckpt_dir $CKPT --device_ids 0 --mean_teacher --max_merge_num 0 --fold 0 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
python $P_P --experiment_name c16_all_num6_label4_0 --ckpt_dir $CKPT --device_ids 0 --fold 0 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &

# python $P_P --experiment_name c16_blank_num6_label4_2 --ckpt_dir $CKPT --device_ids 1 --mean_teacher --ada_pse 0 --max_merge_num 0 --fold 2 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_ada_num6_label4_2 --ckpt_dir $CKPT --device_ids 1 --mean_teacher --max_merge_num 0 --fold 2 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --experiment_name c16_all_num6_label4_2 --ckpt_dir $CKPT --device_ids 1 --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &


# python $P_P --ckpt_dir $CKPT --experiment_name c16_num6_label4_1_shap --device_ids 5 --metrics shap --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
# python $P_P --ckpt_dir $CKPT --experiment_name c16_num6_label4_1_cont --device_ids 6 --metrics cont --fold 1 --max_pseudo_num 6 --pseudo_label_num 4 --ts_dir $LOGDIR &
