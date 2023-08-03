from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
import time
import numpy as np
import random
from pathlib import Path
import re
from parameters import parse_args

# DDP
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

# torch
import torch
import torch.optim as optim

# data_utils
from data_utils import *
from data_utils.tools import read_texts, read_behaviors_text, get_doc_input_bert,read_images,read_behaviors_CV
from data_utils.utils import para_and_log, report_time_train, report_time_eval, save_model_scaler,setuplogger,get_time
from data_utils.dataset import Build_text_CV_Dataset, Build_Lmdb_Dataset,Build_Text_Dataset,Build_Id_Dataset
from data_utils.metrics import get_text_only_scoring, get_itemId_scoring, get_LMDB_only_scoring,get_MMEncoder_scoring,eval_model
from data_utils.lr_decay import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,get_step_schedule_with_warmup

# model
from model.model import Model

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel, BertConfig, BertTokenizer
from transformers import ViTMAEModel,SwinModel, CLIPVisionModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print("BASE_DIR", BASE_DIR)


import hfai

def run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                    model, user_history, users_eval, batch_size, item_num, 
                    mode, is_early_stop, local_rank,args, Log_file, 
                    item_content=None, item_id_to_keys=None):

    eval_start_time = time.time()

    if 'text-only' in args.item_tower:
        Log_file.info('get text-only scoring...')
        item_scoring = get_text_only_scoring(model, item_content, batch_size, args, local_rank)


    if 'CV-only' in args.item_tower:
        Log_file.info('get CV-only scoring...')
        item_scoring = get_LMDB_only_scoring(model, item_num, item_id_to_keys, batch_size, args, local_rank)

            
    if 'modal' in args.item_tower:
        Log_file.info('get Multi-modal (text and CV) scoring...')
        item_scoring = get_MMEncoder_scoring(model, item_content, item_num, item_id_to_keys, batch_size, args, local_rank)

    elif "ID"  in args.item_tower:
        Log_file.info('get ID scoring...')
        item_scoring = get_itemId_scoring(model, item_num, batch_size, args, local_rank)

    valid_Hit10, nDCG10 = eval_model(model, user_history, users_eval, item_scoring, 
                                    batch_size, args,item_num, Log_file, mode, local_rank)

    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 20: 
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break


def run_train_all(local_rank, model_dir,Log_file ,Log_screen, start_time, args):

    # ============================ text and image encoders============================

    if 'modal' in args.item_tower or 'text-only' in args.item_tower:
        if "xlm-roberta-base" in args.bert_model_load or "roberta-base" in args.bert_model_load:
            Log_file.info('load {} model ...'.format(args.bert_model_load))
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR, "..", "TextEncoders", args.bert_model_load))
            tokenizer = AutoTokenizer.from_pretrained(bert_model_load)
            config = AutoConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = AutoModel.from_pretrained(bert_model_load, config=config)

        elif "chinese-roberta-wwm-ext" in args.bert_model_load:
            Log_file.info('load {} model ...'.format(args.bert_model_load))
            bert_model_load = os.path.abspath(os.path.join(BASE_DIR,  "..", "TextEncoders", args.bert_model_load))
            tokenizer = BertTokenizer.from_pretrained(bert_model_load)
            config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = BertModel.from_pretrained(bert_model_load, config=config)

        for index, (name, param) in enumerate(bert_model.named_parameters()):
            # print(index, (name, param.size()))
            if index < args.text_freeze_paras_before:
                param.requires_grad = False

        if 'text-only' in args.item_tower:
            cv_model = None 

    if 'modal' in args.item_tower or 'CV-only' in args.item_tower:

        if "vit-mae-base" in args.CV_model_load:
            Log_file.info('load {} model ...'.format(args.CV_model_load))
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR,  "..", "CVEncoders", args.CV_model_load))
            cv_model = ViTMAEModel.from_pretrained(cv_model_load)

        elif "swin-tiny-patch4-window7-224" in args.CV_model_load or "swin-base-patch4-window7-224" in args.CV_model_load:
            Log_file.info('load {} model ...'.format(args.CV_model_load))
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR,  "..", "CVEncoders", args.CV_model_load))
            cv_model = SwinModel.from_pretrained(cv_model_load)

        elif "clip-vit-base-patch32" in args.CV_model_load:
            Log_file.info('load {} model ...'.format(args.CV_model_load))
            cv_model_load = os.path.abspath(os.path.join(BASE_DIR,  "..", "CVEncoders", args.CV_model_load))
            cv_model = CLIPVisionModel.from_pretrained(cv_model_load, output_hidden_states=True)


        for index, (name, param) in enumerate(cv_model.named_parameters()):
            # print(index, (name, param.size()))
            if index < args.CV_freeze_paras_before:
                param.requires_grad = False

        if 'CV-only' in args.item_tower:
            bert_model = None

    if 'ID' in args.item_tower:
        bert_model = None
        cv_model = None

    
    # exit()
    # ============================ 数据导入 ============================
    item_content = None
    if 'modal' in args.item_tower or 'text-only' in args.item_tower:

        # text可以直接读取每个item的title content到内存
        Log_file.info("Read Item Texts ...")
        item_dic_itme_name_titles_before_match_behaviour, item_name_to_index_before_match_behaviour, item_index_to_name_before_match_behaviour = \
            read_texts(os.path.join(args.root_data_dir, args.dataset, args.texts), args, tokenizer)

        Log_file.info('read behaviors for text ...')
        item_num, item_dic_itme_name_titles_after, item_name_to_index_after, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test = \
            read_behaviors_text(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                        item_dic_itme_name_titles_before_match_behaviour, item_name_to_index_before_match_behaviour,
                        item_index_to_name_before_match_behaviour, args.max_seq_len, args.min_seq_len, Log_file)

        Log_file.info('combine text information...')
        text_title, text_title_attmask = get_doc_input_bert(item_dic_itme_name_titles_after, item_name_to_index_after, args)

        item_content = np.concatenate([x for x in [text_title, text_title_attmask] if x is not None], axis=1)
    
    # image因为占用内存较大，无法直接读取content到内存中
    # 本工作采用lmdb方式对image单独存储，批量读取可以速度很快
    # 在前期对user seq进行处理时候，image与Id的处理方式相似，故将之放在一起
    Log_file.info('read images ...')
    item_dic_name_to_keys_before_match_behaviour, item_name_to_index_before_match_behaviour, item_index_to_name_before_match_behaviour = \
                                            read_images(os.path.join(args.root_data_dir, args.dataset, args.texts))

    Log_file.info('read behaviors for CV and ID...')
    item_num, item_id_to_keys, users_train, users_valid, users_test, \
    users_history_for_valid, users_history_for_test, item_name_to_id, neg_sampling_list, pop_prob_list = \
        read_behaviors_CV(os.path.join(args.root_data_dir, args.dataset, args.behaviors), 
                        item_dic_name_to_keys_before_match_behaviour,
                       item_name_to_index_before_match_behaviour, 
                       item_index_to_name_before_match_behaviour, 
                       args.max_seq_len, args.min_seq_len, Log_file)

    print("users_train", len(users_train))

    # ============================ dataset and dataloader ============================
    if 'modal' in args.item_tower:
        Log_file.info('build text and CV dataset...')
        
        train_dataset = Build_text_CV_Dataset(u2seq=users_train,
                                    item_content=item_content,
                                    max_seq_len=args.max_seq_len,
                                    item_num=item_num,
                                    text_size=args.num_words_title,
                                    db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                    item_id_to_keys=item_id_to_keys,
                                    args=args,
                                    neg_sampling_list=neg_sampling_list) # the neg_sampling_list serve for in_batch and debias
            
    elif 'CV' in args.item_tower:

        train_dataset = Build_Lmdb_Dataset(u2seq=users_train,
                                           item_num=item_num,
                                           max_seq_len=args.max_seq_len,
                                           db_path=os.path.join(args.root_data_dir, args.dataset, args.lmdb_data),
                                           item_id_to_keys=item_id_to_keys, 
                                           args=args,
                                           neg_sampling_list=neg_sampling_list)

    elif "text" in args.item_tower:
        train_dataset = Build_Text_Dataset(userseq=users_train, 
                                           item_content=item_content, 
                                           max_seq_len=args.max_seq_len,
                                           item_num=item_num, 
                                           text_size=args.num_words_title,
                                           args=args,
                                           neg_sampling_list=neg_sampling_list)

    elif "ID" in args.item_tower:
        train_dataset = Build_Id_Dataset(u2seq=users_train, 
                                         item_num=item_num, 
                                         max_seq_len=args.max_seq_len,
                                         args=args,
                                         neg_sampling_list=neg_sampling_list)


    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        multiprocessing_context="fork", #加速
                        worker_init_fn=train_dataset.worker_init_fn, 
                        pin_memory=True, 
                        sampler=train_sampler)

    # ============================ 模型 ============================
    Log_file.info('build model...')
    model = Model(args, item_num, bert_model, cv_model, pop_prob_list).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    

    if not hfai.client.receive_suspend_command():
        try:
            epoches = []
            for file in os.listdir(model_dir):
                epoches.append(int(re.split(r'[._-]', file)[1]))
            args.load_ckpt_name = 'epoch-' + str(max(epoches)) + '.pt'
            ckpt_path = os.path.abspath(os.path.join(model_dir, args.load_ckpt_name))
            start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
            print("start_epoch", start_epoch)
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            Log_file.info('load checkpoint...')
            # model.load_state_dict(checkpoint['model_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) #也可以迁移单模态
            Log_file.info(f"Model loaded from {args.load_ckpt_name}")
            torch.set_rng_state(checkpoint['rng_state'])  # load torch的随机数生成器状态
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # load torch.cuda的随机数生成器状态
            is_early_stop = False

        except Exception as e:
            ckpt_path = args.pretrain_load_ckpt_name
            start_epoch = 0
            print("pre-train start_epoch", start_epoch)
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            Log_file.info('load pre-train checkpoint...')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) #也可以迁移单模态
            is_early_stop = False


    Log_file.info('model.cuda()...')
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # ============================ 优化器 ============================

    image_net_params = []
    bert_params = []
    recsys_params = []

    layer_recsys = 0
    layer_text = 0
    layer_cv = 0
    for index, (name, param) in enumerate(model.module.named_parameters()):
        if param.requires_grad:
            if 'cv_encoder' in name:
                if 'cv_proj' in name:
                    recsys_params.append(param)
                    # layer_recsys += 1
                    # print(layer_recsys, name)
                else:
                    image_net_params.append(param)
                    # layer_cv += 1
                    # print(layer_cv, name)
            elif "text_encoder" in name:
                if "text_proj" in name:
                    recsys_params.append(param)
                    # layer_recsys += 1
                    # print(layer_recsys, name)
                else:
                    bert_params.append(param)
                    # layer_text += 1
                    # print(layer_text, name)
            else:
                recsys_params.append(param)
                # layer_recsys += 1
                # print(layer_recsys, name)


    optimizer = optim.AdamW([
        {'params': bert_params,'lr': args.text_fine_tune_lr, 'weight_decay': 0,  'initial_lr': args.text_fine_tune_lr},
        {'params': image_net_params, 'lr': args.CV_fine_tune_lr, 'weight_decay': 0, 'initial_lr': args.CV_fine_tune_lr},
        {'params': recsys_params, 'lr': args.lr, 'weight_decay': args.weight_decay, 'initial_lr': args.lr}
        ])
    

    Log_file.info("***** {} finetuned parameters in text encoder *****".format(
        len(list(bert_params))))
    Log_file.info("***** {} fiuetuned parameters in image encoder*****".format(
        len(list(image_net_params))))
    Log_file.info("***** {} parameters with grad in recsys *****".format(
        len(list(recsys_params))))


    model_params_require_grad = []
    model_params_freeze = []
    for param_name, param_tensor in model.module.named_parameters():
        if param_tensor.requires_grad:
            model_params_require_grad.append(param_name)
        else:
            model_params_freeze.append(param_name)

    Log_file.info("***** model: {} parameters require grad, {} parameters freeze *****".format(
        len(model_params_require_grad), len(model_params_freeze)))


    if start_epoch != 0: # load 优化器状态 for 断点重续,如果是迁移学习，是不会导入优化器的
        optimizer.load_state_dict(checkpoint["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path}")

    # ============================  训练 ============================

    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    Log_screen.info('{} train start'.format(args.label_screen))

    #使用混合精度
    from torch.cuda.amp import autocast as autocast
    scaler = torch.cuda.amp.GradScaler()

    ## load scaler的状态
    if start_epoch != 0: # load 半精度的状态 for 断点重续, 如果是迁移学习，是不会导入的
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")


    # lr dacay, 本文暂时用不上
    warmup_steps = 0  # 设置成0，没有warm_up
    if args.scheduler == "cosine_schedule":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps, # 设置成0，没有warm_up
            num_training_steps=args.epoch,
            start_epoch=start_epoch)
        
    elif args.scheduler == "linear_schedule":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps, # 设置成0，没有warm_up
            num_training_steps=args.epoch,
            start_epoch=start_epoch)
        
    elif args.scheduler == "step_schedule":
        lr_scheduler = get_step_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps, # 设置成0，没有warm_up
            gap_steps = args.scheduler_steps, #每N轮下降10倍
            start_epoch=start_epoch)
    else:
        lr_scheduler = None
        

    epoch_left = args.epoch - start_epoch
    for ep in range(epoch_left):
        now_epoch = start_epoch + ep + 1
        train_dl.sampler.set_epoch(now_epoch)
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        loss, batch_index, need_break = 0.0, 1, False
        model.train()

        if lr_scheduler is not None:
            Log_file.info('start of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        loss = 0
        loss_rec = 0
        loss_i = 0
        align, uniform = 0, 0
        Log_file.info('temprature {} in epoch {} '.format(model.module.temp, now_epoch))
        for data in train_dl:
            if 'modal' in args.item_tower:
                sample_items_id, sample_items_text, sample_items_CV, log_mask = data
                sample_items_id, sample_items_text, sample_items_CV,log_mask = sample_items_id.to(local_rank), sample_items_text.to(local_rank), \
                                                              sample_items_CV.to(local_rank),log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_CV = sample_items_CV.view(-1, 3, args.CV_resize, args.CV_resize)
                sample_items_id = sample_items_id.view(-1)


            elif 'text' in args.item_tower:
                sample_items_id, sample_items_text, log_mask = data
                sample_items_id, sample_items_text, log_mask = sample_items_id.to(local_rank), sample_items_text.to(local_rank), log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_CV = None
                sample_items_id = sample_items_id.view(-1)

            elif 'CV' in args.item_tower:
                sample_items_id, sample_items_CV, log_mask = data
                sample_items_id, sample_items_CV, log_mask = sample_items_id.to(local_rank), sample_items_CV.to(local_rank), log_mask.to(local_rank)
                sample_items_CV =  sample_items_CV.view(-1, 3, args.CV_resize, args.CV_resize)
                sample_items_text = None
                sample_items_id = sample_items_id.view(-1)

            elif "ID" in args.item_tower:
                sample_items_id, log_mask = data
                sample_items_id, log_mask = sample_items_id.to(local_rank), log_mask.to(local_rank)
                sample_items_id = sample_items_id.view(-1)
                sample_items_text = None
                sample_items_CV = None

            optimizer.zero_grad()

            # mix 精度
            with autocast(enabled=True):
                bz_loss, bz_loss_rec, bz_loss_i,  bz_align, bz_uniform = model(sample_items_id,sample_items_text, sample_items_CV, log_mask, local_rank, args)
                loss += bz_loss.data.float()
                loss_rec += bz_loss_rec.data.float()
                loss_i += bz_loss_i.data.float()
                align += bz_align.data.float()
                uniform += bz_uniform.data.float()
                

            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 断点重续
            if dist.get_rank() == 0 and hfai.client.receive_suspend_command():
                hfai.client.go_suspend()

            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:

                # Log_file.info('cnt: {}, Ed: {}, batch loss_all: {:.5f}, loss_rec : {:.5f}, loss_II : {:.5f}'.format(
                #     batch_index, batch_index * args.batch_size, loss.data / batch_index, loss_rec.data / batch_index, loss_i.data / batch_index))
                Log_file.info('cnt: {}, Ed: {}, batch loss_all: {:.5f}, loss_rec : {:.5f}, loss_II : {:.5f}, align : {:.5f}, uniform : {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss_rec.data / batch_index, loss_i.data / batch_index,  align.data / batch_index, uniform.data / batch_index))

            batch_index += 1



        if not need_break and now_epoch % 1 == 0:

            # valid
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break = \
                run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                        model, users_history_for_valid, users_valid, args.batch_size, item_num, 
                        args.mode, is_early_stop, local_rank, args, Log_file, item_content,item_id_to_keys)
            
            # test，因为inbatch花不了多少时间，干脆就直接顺便测试了
            run_eval_all(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                        model, users_history_for_test, users_test, args.batch_size, item_num,
                        args.mode, is_early_stop, local_rank, args, Log_file, item_content,item_id_to_keys)

            model.train()

            if dist.get_rank() == 0:
                epoches = []
                for file in os.listdir(model_dir):
                    epoches.append(file)

                Log_file.info(' Delete pt except for saving memory ...')
                for file in epoches:
                    suffix = int(re.split(r'[._-]', file)[1])
                    if suffix % 5 != 0 and max_epoch != suffix: #只保留5的倍数对应的权重(以防万一中途打断后找不到)和max_epoch的值
                        os.remove(os.path.join(model_dir, file))

        if dist.get_rank() == 0:
            save_model_scaler(now_epoch, model, model_dir, scaler, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file) # mix

        if lr_scheduler is not None:
            lr_scheduler.step()
            Log_file.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_file.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))

        if need_break:
            break


    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# def main(local_rank): # hfai的多机多卡通信方式（也可以使用一个node）
# # 切分成单机多卡使用
def main():

    args = parse_args()
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # 多机通信
#     ip = os.environ['MASTER_ADDR']
#     port = os.environ['MASTER_PORT']
#     hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
#     rank = int(os.environ['RANK'])  # 当前机器编号
#     gpus = torch.cuda.device_count()  # 每台机器的GPU个数

#     # world_size是全局GPU个数，rank是当前GPU全局编号
#     dist.init_process_group(backend='nccl',
#                             init_method=f'tcp://{ip}:{port}',
#                             world_size=hosts * gpus,
#                             rank=rank * gpus + local_rank)
#     torch.cuda.set_device(local_rank)
#     args.node_rank = rank # 为多机多卡准备
    
    args.local_rank = local_rank # 添加
    args.node_rank = 0 # 为单机机多卡准备

    setup_seed(123456)  # 设置固定的随机种子
    
    # 导入的模型
    pretrain_name = args.pretrain_load_ckpt_name.split("checkpoint")[1].split("_")[1]

    #记录
    if 'modal' in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
                    f"_ed_{args.embedding_dim}_lr_{args.lr}" \
                    f"_FlrText_{args.text_fine_tune_lr}_FlrImg_{args.CV_fine_tune_lr}"\
                    f"_{args.bert_model_load}_{args.CV_model_load}" \
                    f'_freeze_{args.text_freeze_paras_before}_{args.CV_freeze_paras_before}'\
                    f"_len_{args.max_seq_len}" \
                    f"_{args.fusion_method}"\
                    f"_{args.benchmark}"\
                    f"_{args.scheduler}{args.scheduler_steps}"\
                    f"_UI_alpha{args.UI_alpha}"\
                    f"_UI_temp{args.UI_temp}"\
                    f"_transfer_{pretrain_name}"

    elif "text-only" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower) 
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_FlrText_{args.text_fine_tune_lr}"\
            f"_{args.bert_model_load}"\
            f'_freeze_{args.text_freeze_paras_before}'\
            f"_len_{args.max_seq_len}" \
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"\
            f"_transfer_{pretrain_name}"


    elif "CV-only" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_FlrImg_{args.CV_fine_tune_lr}"\
            f"_{args.CV_model_load}"\
            f'_freeze_{args.CV_freeze_paras_before}'\
            f"_len_{args.max_seq_len}"\
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"\
            f"_transfer_{pretrain_name}"

    elif "ID" in args.item_tower:
        dir_label =  str(args.behaviors).strip().split(".")[0] + "_"  + str(args.item_tower)
        log_paras = f"bs{args.batch_size}" \
            f"_ed_{args.embedding_dim}_lr_{args.lr}" \
            f"_len_{args.max_seq_len}"\
            f"_{args.benchmark}"\
            f"_{args.scheduler}{args.scheduler_steps}"


    model_dir = os.path.join("./checkpoint_" + dir_label, f"cpt_" + log_paras)
    time_run = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)
    Log_file.info(time_run) # 写在log文件里面，否则容易被打断后产生过多片段

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if "train" in args.mode:
        run_train_all(local_rank, model_dir,Log_file ,Log_screen, start_time, args)

    end_time = time.time()
    hour, minute, seconds = get_time(start_time, end_time)
    Log_file.info("#### (time) all: hours {} minutes {} seconds {} ####".format(hour, minute, seconds))


if __name__ == '__main__':
    # ngpus = torch.cuda.device_count()
    # torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
    # # 切分使用
    main()

