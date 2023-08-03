import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR,"..",".."))


# 选择下游数据集

f=open('dataset_path.txt','r')


lines=f.readlines()  #读取整个文件所有行，保存在 list 列表中

dataset_list = []
behaviors_list = []
texts_list = []
lmdb_data_list = []
for line in lines:
    splited = line.strip().split(",")
    if len(splited) > 1:
        if splited[0] in "dataset_list":
            dataset_list.append(splited[1])
        if splited[0] in "behaviors_list":
            behaviors_list.append(splited[1])
        if splited[0] in "texts_list":
            texts_list.append(splited[1])
        if splited[0] in "lmdb_data_list":
            lmdb_data_list.append(splited[1])
    # print(splited)  # 默认换行
    line = f.readline()

data_index_list = [i for i in range(len(lmdb_data_list))]


# 选择上游预训练模型
pretrain_load_ckpt_name = '/ceph-jd/pub/jupyter/yuanfj/notebooks/liyouhua/Recsys_Inbatch_OGM_MM/V1_27_merge_next_II_cut_user/checkpoint_kuaishou_pick_users_20W_modal/cpt_bs64_ed_768_lr_0.0001_FlrText_0.0001_FlrImg_0.0001_xlm-roberta-base_clip-vit-base-patch32_freeze_165_165_len_20_merge_attn_sasrec_step_schedule1000_UI_alpha0.4_UI_temp0.1/epoch-31.pt'

logging_num = 4
testing_num = 1

CV_resize = 224

CV_model_load = "clip-vit-base-patch32"


BERT_model_load =  "xlm-roberta-base"

CV_freeze_paras_before = 165 #mae: 164, swin-T: 183, swin-B: 387, clip:165
text_freeze_paras_before = 165 

CV_fine_tune_lr = 1e-4
text_fine_tune_lr = 1e-4


mode = 'train' # train test
item_tower = 'modal'  # modal, text-only, CV-only, ID

epoch = 30

l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [64]
lr_list = [1e-4]
embedding_dim_list = [768]
max_seq_len_list = [20]


benchmark_list = ['sasrec'] # 'sasrec', 'grurec', 'nextit'
fusion_method = "merge_attn" # ['co_att', 'merge_attn','sum', 'concat', 'film', 'gated']

scheduler_steps = 1000


for weight_decay in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for lr in lr_list:
                    for max_seq_len in max_seq_len_list:
                            for benchmark in benchmark_list:
                                for data_index in data_index_list:
                                    dataset = dataset_list[data_index]
                                    behaviors = behaviors_list[data_index]
                                    texts = texts_list[data_index]
                                    lmdb_data = lmdb_data_list[data_index]

                                    label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                                            item_tower, batch_size, embedding_dim, lr,
                                            drop_rate, weight_decay, max_seq_len)

                                    run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                                            python  -m torch.distributed.launch --nproc_per_node 8 --master_port 1251  run-single.py\
                                            --root_data_dir {}  --dataset {} --behaviors {} --texts {}  --lmdb_data {}\
                                            --mode {} --item_tower {} --pretrain_load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                            --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                                            --CV_resize {} --CV_model_load {} --bert_model_load {}  --epoch {} \
                                            --text_freeze_paras_before {} --CV_freeze_paras_before {} --max_seq_len {} \
                                            --CV_fine_tune_lr {} --text_fine_tune_lr {} --fusion_method {} --benchmark {} --scheduler_steps {}".format(
                                        root_data_dir, dataset, behaviors, texts, lmdb_data,
                                        mode, item_tower, pretrain_load_ckpt_name, label_screen, logging_num, testing_num,
                                        weight_decay, drop_rate, batch_size, lr, embedding_dim,
                                        CV_resize, CV_model_load, BERT_model_load, epoch,
                                        text_freeze_paras_before, CV_freeze_paras_before, max_seq_len,
                                        CV_fine_tune_lr, text_fine_tune_lr, fusion_method, benchmark, scheduler_steps)

                                    os.system(run_py)










