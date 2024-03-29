nohup python -m torch.distributed.launch --nproc_per_node 2 main.py \
--train_data_path data/dataset_emoji_train_data.csv \
--val_data_path data/dataset_emoji_val_data.csv \
--test_data_path data/dataset_emoji_test_data.csv \
--model Roberta \
--gpus 2 \
--lr 5e-5 \
--epochs 3 \
--batch_size 512 \
--noise EDA \
--prob 0.3 \
--adv_step 3 \
--num_labels 20 \
--tasks_kinds Classification &