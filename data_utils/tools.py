
import torch
import numpy as np
import tqdm


def read_images(images_path):
    item_dic_name_to_keys = {} # keys指lmdb文件中的索引
    item_name_to_index = {}
    item_index_to_name = {}
    index = 1
    with open(images_path, "r") as f:
        # for line in tqdm.tqdm(f):
        for line in f:
            splited = line.strip('\n').split('\t')
            image_name = splited[0]
            item_name_to_index[image_name] = index
            item_index_to_name[index] = image_name
            item_dic_name_to_keys[index] = u'{}'.format(int(image_name.replace('v', ''))).encode('ascii')
            index += 1
    return item_dic_name_to_keys, item_name_to_index, item_index_to_name


def read_behaviors_CV(behaviors_path, before_item_id_to_keys, before_item_name_to_index, before_item_index_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### images number {} {} (before matching with user seqs)#####".
                  format(len(before_item_id_to_keys), len(before_item_name_to_index)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_index)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        # for line in tqdm.tqdm(f):
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            user_seqs_by_item_name = splited[1].split(' ')
            if len(user_seqs_by_item_name) < min_seq_len:
                continue
            user_seqs_by_item_name = user_seqs_by_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_index[i] for i in user_seqs_by_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs before matching with user seqs {}".format(pairs_num))
    Log_file.info("##### user seqs before matching with user seqs {}".format(before_seq_num))

    item_id = 1
    item_id_to_keys = {}
    item_id_before_to_now = {}
    item_name_to_id = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_keys[item_id] = before_item_id_to_keys[before_item_id]
            item_name_to_id[before_item_index_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after matching with user seqs {}, {}, {}, {}#####".format(item_num, item_id - 1, len(item_id_to_keys), len(item_id_before_to_now)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    neg_sampling_list = []
    train_item_counts = [0] * (item_num + 1)
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        users_train[user_id] = user_seq[:-2]
        users_valid[user_id] = user_seq[-(max_seq_len+2):-1]
        users_test[user_id] = user_seq[-(max_seq_len+1):]

        for i in users_train[user_id]:
            train_item_counts[i] += 1

        for i in user_seq:
            neg_sampling_list.append(i)
        users_history_for_valid[user_id] = torch.LongTensor(np.array(users_train[user_id]))
        users_history_for_test[user_id] = torch.LongTensor(np.array(users_valid[user_id]))
        user_id += 1
        
    # for debias
    item_counts_powered = np.power(train_item_counts, 1.0)
    pop_prob_list = []
    for i in range(1, item_num + 1):
        pop_prob_list.append(item_counts_powered[i])
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)
        
    Log_file.info("##### user seqs after matching with user seqs {}, {}, {}, {}, {} interaction num {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid),
                         len(users_test), len(neg_sampling_list)))

    return item_num, item_id_to_keys, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id, neg_sampling_list,pop_prob_list


def read_texts(text_path, args, tokenizer):
    item_dic = {}
    item_name_to_index = {}
    item_index_to_name = {}
    index = 1
    with open(text_path, "r") as f:
        # for line in tqdm.tqdm(f):
        for line in f:
            splited = line.strip('\n').split('\t')
            item_name, title = splited
        
            item_name_to_index[item_name] = index
            item_index_to_name[index] = item_name
            index += 1
            # tokenizer
            title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
            item_dic[item_name] = [title]

    return item_dic, item_name_to_index, item_index_to_name


def read_behaviors_text(behaviors_path, item_dic, before_item_name_to_index, before_item_index_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### items of text {}, {}, and {} (before matching with user seqs) #####".
                  format(len(before_item_name_to_index), len(item_dic), len(before_item_index_to_name)))

    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_index)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
        
    Log_file.info('rebuild user seqs ...')
    with open(behaviors_path, "r") as f:
        # for line in tqdm.tqdm(f):
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            user_seqs_by_item_name = splited[1].split(" ")

            if len(user_seqs_by_item_name) < min_seq_len:
                continue

            user_seqs_by_item_name = user_seqs_by_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_index[i] for i in user_seqs_by_item_name] 
            user_seq_dic[user_id] = user_seqs_by_item_name
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1

    Log_file.info("##### pairs before matching with user seqs {}".format(pairs_num))
    Log_file.info("#### user seqs before matching with user seqs {}".format(before_seq_num))

    for item_id in range(1, before_item_num + 1):
        if before_item_counts[item_id] == 0:
            item_dic.pop(before_item_index_to_name[item_id])
    
    item_dic_after = item_dic

    item_id = 1
    item_num = len(item_dic_after)
    item_index = {}

    for item_name, item_title in item_dic_after.items():
        item_index[item_name] = item_id
        item_id += 1

    Log_file.info("##### items after matching with user seqs {}, {}#####".format(item_num, len(item_index)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, item_name_list in user_seq_dic.items():
        
        user_seq = [item_index[item_name] for item_name in item_name_list]  
        
        users_train[user_id] = user_seq[:-2]
        users_valid[user_id] = user_seq[-(max_seq_len+2):-1]
        users_test[user_id] = user_seq[-(max_seq_len+1):]
        
        users_history_for_valid[user_id] = torch.LongTensor(np.array(users_train[user_id])) 
        users_history_for_test[user_id] = torch.LongTensor(np.array(users_valid[user_id])) #用于计算score使用，提前生成，免得届时再生成

        user_id += 1

    Log_file.info("##### user seqs after matching with user seqs {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_dic_after, item_index, users_train, users_valid, users_test, users_history_for_valid, users_history_for_test


def get_doc_input_bert(news_dic, item_index, args):
    item_num = len(news_dic) + 1

    news_title = np.zeros((item_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')

    # for key in tqdm.tqdm(news_dic):
    for key in news_dic:
        
        title = news_dic[key]
        doc_index = item_index[key]
        
        news_title[doc_index] = title[0]['input_ids'] # values来自于tokenizer后形成的值
        news_title_attmask[doc_index] = title[0]['attention_mask'] # values来自于tokenizer后形成的值

    return news_title, news_title_attmask
