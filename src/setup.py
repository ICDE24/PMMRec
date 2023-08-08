import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR,".."))


# select downstream dataset

dataset = 'bili_food'
behaviors = 'bilibili_food_users.tsv'
texts = 'bilibili_food_items_texts.tsv'
lmdb_data = 'bilibili_food_items_images.lmdb'

# PMMRec pre-training model
pretrain_load_ckpt_name = '../pretrain-PT/epoch-0.pt' 
pretrain_part = "item_encoder" # 'full', 'item_encoder', 'user_encoder'

logging_num = 4
testing_num = 1

CV_resize = 224

CV_model_load = "clip-vit-base-patch32"
BERT_model_load =  "xlm-roberta-base"

CV_freeze_paras_before = 165  # clip-vit:165, only finetune top 2 transformer blocks
text_freeze_paras_before = 165  # Roberta-multi-lingual: 165, only finetune top 2 transformer blocks

CV_fine_tune_lr = 1e-4 
text_fine_tune_lr = 1e-4 


mode = 'transfer' # Train-from-scratch (TFS), transfer
item_tower = 'modal'  # modal, text-only, CV-only, corresponding to multi-modal and two single-modal settings

epoch =  5 # For transferring, no more than 10 epochs are enough

l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [32]
lr_list = [1e-4]
embedding_dim_list = [768]
max_seq_len_list = [20]


benchmark_list = ['sasrec'] # 'sasrec', 'grurec', 'nextit' #Choose a different order to recommend the architecture
fusion_method = "merge_attn" # ['co_att', 'merge_attn','sum', 'concat', 'film', 'gated'] #Different multimodal fusion methods

scheduler_steps = 1000


for weight_decay in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for lr in lr_list:
                    for max_seq_len in max_seq_len_list:
                            for benchmark in benchmark_list:

                                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                                        item_tower, batch_size, embedding_dim, lr,
                                        drop_rate, weight_decay, max_seq_len)

                                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                                        python  -m torch.distributed.launch --nproc_per_node 8 --master_port 1251  main.py\
                                        --root_data_dir {}  --dataset {} --behaviors {} --texts {}  --lmdb_data {}\
                                        --mode {} --item_tower {} --pretrain_load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                        --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                        --CV_resize {} --CV_model_load {} --bert_model_load {}  --epoch {} \
                                        --text_freeze_paras_before {} --CV_freeze_paras_before {} --max_seq_len {} \
                                        --CV_fine_tune_lr {} --text_fine_tune_lr {} --fusion_method {} --benchmark {} --scheduler_steps {} --pretrain_part {}".format(
                                    root_data_dir, dataset, behaviors, texts, lmdb_data,
                                    mode, item_tower, pretrain_load_ckpt_name, label_screen, logging_num, testing_num,
                                    weight_decay, drop_rate, batch_size, lr, embedding_dim,
                                    CV_resize, CV_model_load, BERT_model_load, epoch,
                                    text_freeze_paras_before, CV_freeze_paras_before, max_seq_len,
                                    CV_fine_tune_lr, text_fine_tune_lr, fusion_method, benchmark, scheduler_steps, pretrain_part)

                                os.system(run_py)










