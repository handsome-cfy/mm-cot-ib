# conda activate mmcot
beta=0.001
device=0
echo $beta
CUDA_VISIBLE_DEVICES=$device python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_rationale --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 --final_eval --prompt_format QCM-LEA --epoch 20 --vot_num 0 --alpha 0.5 --beta $beta --output_dir ./results/base_train_from_scratch/beta_$beta

CUDA_VISIBLE_DEVICES=$device python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_rationale --final_eval --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 --prompt_format QCM-LEA --beta $beta --evaluate_dir ./results/base_train_from_scratch/beta_$beta/mmcot_ib_rationale
