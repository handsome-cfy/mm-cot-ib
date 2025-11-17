#!/usr/bin/env bash
# TRAIN - rationale generation
CUDA_VISIBLE_DEVICES=3 python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_rationale --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 --final_eval --prompt_format QCM-LE --epoch 20 --vot_num 5 --alpha 0.5 --output_dir ./results/base_train_from_scratch
# EVAL - rationale generation
CUDA_VISIBLE_DEVICES=3 python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_rationale --final_eval --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 512 --prompt_format QCM-LE --evaluate_dir ./results/base_train_from_scratch/mmcot_ib_rationale
# Train - answer inference
CUDA_VISIBLE_DEVICES=3 python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_answer --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --epoch 20 --vot_num 5 --alpha 0.5 --eval_le ./results/base_train_from_scratch/mmcot_ib_rationale/predictions_ans_eval.json --test_le ./results/base_train_from_scratch/mmcot_ib_rationale/predictions_ans_test.json --output_dir ./results/base_train_from_scratch
# EVAL - answer inference
CUDA_VISIBLE_DEVICES=3 python main.py --model ./models/unifiedqa-t5-base --user_msg mmcot_ib_answer --img_type clip --bs 1 --eval_bs 1 --eval_acc 10 --output_len 64 --final_eval --prompt_format QCMG-A --eval_le ./results/base_train_from_scratch/mmcot_ib_rationale/predictions_ans_eval.json --test_le ./results/base_train_from_scratch/mmcot_ib_rationale/predictions_ans_test.json --evaluate_dir ./results/base_train_from_scratch/mmcot_ib_answer