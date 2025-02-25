import numpy as np
import pandas as pd  
import torch
import torch.nn.functional as F
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
# import deepctr_torch as deepctr
from deepctr_torch.inputs import SparseFeat,DenseFeat,VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from SourceCode.torch_youtubednn import YouTubeDNN

warnings.filterwarnings('ignore')

# 设置随机种子
def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 采样数据
def get_all_click_sample(data_path, sample_nums=50000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click
# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click
# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df
# 读取文章的Embedding数据
def get_item_emb_dict(data_path,resume=False):
    if resume and os.path.exists(save_path + 'item_content_emb.pkl'):
        with open(save_path + 'item_content_emb.pkl', 'rb') as file:
            item_emb_dict = pickle.load(file)
            return item_emb_dict
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))
    
    return item_emb_dict
# 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
def get_user_item_time(click_df):

    click_df = click_df.sort_values('click_timestamp')  # 按时间排序
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict
# 根据时间获取item被点击的用户序列  {item1: {user1: time1, user2: time2...}...}
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))
    
    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    
    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    
    return item_type_dict, item_words_dict, item_created_time_dict
# 获取用户历史点击的文章信息
def get_user_hist_item_info_dict(all_click):
    
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))
    
    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))
    
    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))
    
    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
    
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)
    
    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))
    
    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict
# 获取总点击最多的文章  Todo: 可以改成根据user最后点击时间召回近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click
# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)
# YoutubeDNN召回
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()
        
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))   # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)  # 对于每个正样本，选择n个负样本
            
        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0],1,len(pos_list)))
            
        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):   # 对从2开始的每个位置都根据前面的预测
            hist = pos_list[:i]
            
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist)))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist))) # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist)))
                
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set
# item_emb
def gen_data_set_plus(data,item_emb,item_index_2_rawid, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()
        
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))   # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)  # 对于每个正样本，选择n个负样本
            
        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0],item_emb[item_index_2_rawid[pos_list[0]]],1,len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0],item_emb[item_index_2_rawid[pos_list[0]]],1,len(pos_list)))
            
        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):   # 对从2开始的每个位置都根据前面的预测
            hist = pos_list[:i]
            
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i],item_emb[item_index_2_rawid[pos_list[i]]], 1, len(hist)))  # 正样本 [user_id, his_item, pos_item,pos_item_emb, label, len(his_item)]
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi],item_emb[item_index_2_rawid[neg_list[i*negsample+negi]]], 0,len(hist))) # 负样本 [user_id, his_item, neg_item,neg_item_emb, label, len(his_item)]
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i],item_emb[item_index_2_rawid[pos_list[i]]],1,len(hist)))
                
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set
# 将输入的数据进行padding，使得序列特征的长度都一致
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if maxlen is None:
        maxlen = np.max([len(s) for s in sequences])
 
    # 循环处理每个序列
    padded_sequences = []
    for seq in sequences:
        # 截断序列
        if truncating == 'pre':
            trunc_seq = seq[-maxlen:]
        else:
            trunc_seq = seq[:maxlen]
        
        # 填充序列
        if padding == 'pre':
            padded_seq = [value] * (maxlen - len(trunc_seq)) + trunc_seq
        else:
            padded_seq = trunc_seq + [value] * (maxlen - len(trunc_seq))
        
        # 添加到结果列表
        padded_sequences.append(padded_seq)
    
    # 返回结果数组
    return np.array(padded_sequences, dtype=dtype)

def gen_model_input(train_set,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_click_article_emb = np.array([line[3] for line in train_set])
    train_label = np.array([line[4] for line in train_set])
    train_hist_len = np.array([line[5] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid,'click_article_emb':train_click_article_emb
        , "hist_article_id": train_seq_pad,"hist_len": train_hist_len}

    return train_model_input, train_label



# 时序信息加位置编码不行 不止使用用户点击序列 使用item点击序列
def youtubednn_u2i_dict(data,item_emb_df, topk=20, resume=False,max_epochs=1,negsample=0):
    if resume and os.path.exists(save_path + 'youtube_u2i_dict.pkl'):
        with open(save_path + 'youtube_u2i_dict.pkl', 'rb') as file:
            user_recall_items_dict = pickle.load(file)
        return user_recall_items_dict

    # 处理emb
    # item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    emb_dim = len(item_emb_cols)
    # sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')
    print('user_profile_',user_profile_[:10])
    print('item_profile_',item_profile_[:10])
    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}  # 其实就是这个特征的vocabulary_size

    for feature in features:
        lbe = LabelEncoder()  # 把不连续的id编码成从0开始的连续label
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates('user_id')  # user_id已经被LabelEncoder改变了
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本

    # 已经存好了
    # train_set, test_set = gen_data_set_plus(data,item_emb_np,item_index_2_rawid,negsample=negsample)
    # pickle.dump(train_set, open(save_path + '2_tower_train_set.pkl', 'wb'))
    # pickle.dump(test_set, open(save_path + '2_tower_test_set.pkl', 'wb'))
    with open(save_path + '2_tower_train_set.pkl', 'rb') as file:
        train_set = pickle.load(file)
    with open(save_path + '2_tower_test_set.pkl', 'rb') as file:
        test_set = pickle.load(file)
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    # 确定Embedding的维度
    embedding_dim = 16

    # user_embs_0 = user_embs[0]
    # user_0_id = user_index_2_rawid[3]
    # user_his_items = data[data['user_id']==user_0_id]["click_article_id"]
    # print('user_his_items',user_his_items)
    # exit()
    # print('train_set[0]',train_set[0])

    
    # 将数据整理成模型可以直接输入的形式
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                           embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim),
                            DenseFeat('click_article_emb',emb_dim)]

    # 模型的定义
    # num_sampled: 负采样时的样本数量
    linear_feature_columns = user_feature_columns + item_feature_columns
    model = YouTubeDNN(user_feature_columns, item_feature_columns,
                       dnn_hidden_units=(64, embedding_dim), optimizer='Adam',)


    # 模型编译
    # model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    # batch_size=256
    
    history = model.fit(train_model_input, train_label, batch_size=256, max_epochs=max_epochs)

    
    
    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    test_user_model_input = test_model_input
    all_item_emb = item_emb_np[item_profile_['click_article_id'].values]
    print('all_item_emb.shape',all_item_emb.shape)
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values,'click_article_emb':all_item_emb}   #

    model.mode = "user_representation"
    user_embs = model.full_predict(test_user_model_input, batch_size=2 ** 12).squeeze()
    model.mode = "item_representation"
    item_embedding_model = model.rebuild_feature_index(item_feature_columns)
    item_embs = item_embedding_model.full_predict(all_item_model_input, batch_size=2 ** 12)  # item_embs不全覆盖所有item，不会使用


    user_emb_14326 = user_embs[14326]
    # user_raw_id = user_index_2_rawid[14326]
    item_emb_5837 = item_embs[5837]
    item_emb_9046 = item_embs[9046]
    print('after ',F.cosine_similarity(user_emb_14326, item_emb_5837, dim=-1))
    print('after ',F.cosine_similarity(user_emb_14326, item_emb_9046, dim=-1))
    # user_his_items = data[data['user_id']==user_0_id]["click_article_id"]
    # print('user_his_items',user_his_items)
    exit()
    print(user_embs.shape)
    item_embs = item_embs.squeeze()
    print(item_embs.shape)
    # exit()
    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # # 将Embedding转换成字典的形式方便查询  耗时内存大
    # raw_user_id_emb_dict = {user_index_2_rawid[k]: \
    #                             v for k, v in zip(user_profile['user_id'], user_embs)}
    # raw_item_id_emb_dict = {item_index_2_rawid[k]: \
    #                             v for k, v in zip(item_profile['click_article_id'], item_embs)}
    # # 将Embedding保存到本地
    # pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_youtube_emb.pkl', 'wb'))
    # pickle.dump(raw_item_id_emb_dict, open(save_path + 'item_youtube_emb.pkl', 'wb'))

    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    # res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.IndexFlatIP(embedding_dim)
    # gpu_index = faiss.index_cpu_to_all_gpus(index)
    # 上面已经进行了归一化，这里可以不进行归一化了
    #     faiss.normalize_L2(user_embs)
    #     faiss.normalize_L2(item_embs)
    item_embs = item_embs.numpy()
    index.add(item_embs)  # 将item向量构建索引
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)  # 通过user去查询最相似的topk个item

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # user找item 没必要去掉1
        for rele_idx, sim_value in zip(rele_idx_list, sim_value_list):
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                     .get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}  # Todo sorted(v.items(), key=lambda x: x[1], reverse=True)好像有bug
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(save_path + 'youtube_u2i_dict.pkl', 'wb'))
    return user_recall_items_dict
# 使用item点击序列


# In[ ]:


# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来


import argparse

if __name__ == "__main__":
    data_path = './data/'
    save_path = './temp_results/'
    # 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seed_everything()
    metric_recall = True

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max_epochs', type=int, default=1,
                        help='Number of maximum epochs (default: 10)')
    parser.add_argument('--negsample', type=int, default=0,
                        help='Number of negative samples (default: 5)')
    args = parser.parse_args()
    max_epochs = args.max_epochs
    negsample = args.negsample
    print("Max Epochs:", max_epochs)
    print("Negative Samples:", negsample)

    item_emb_df = pd.read_csv(data_path + '/articles_emb.csv')
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    # 采样数据
    all_click_df = get_all_click_sample(data_path,sample_nums=10000)
    # 全量训练集
    # all_click_df = get_all_click_df(offline=False)
    # 对时间戳进行归一化,用于在关联规则的时候计算权重
    all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
    # 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    # 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
    user_multi_recall_dict = {'itemcf_sim_itemcf_recall': {},
                              'usercf_sim_itemcf_recall': {},
                              'embedding_sim_item_recall': {},
                              'youtubednn_recall': {},
                              'youtubednn_usercf_recall': {},
                              'cold_start_recall': {}}

    if not metric_recall:
        user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, item_emb_df, topk=20,
                                                                          resume=False,max_epochs=max_epochs,
                                                                          negsample=negsample)
    else:
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(
            all_click_df)  # trn_hist_click_df 没有最后点击的item 召回评估时不是所有item都有embedding
        # print(len(all_click_df['click_article_id'].unique()))
        # print(len(trn_hist_click_df['click_article_id'].unique()))
        # print(len(trn_last_click_df['click_article_id'].unique()))
        user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, item_emb_df, topk=20,
                                                                          resume=False,max_epochs=max_epochs,
                                                                          negsample=negsample)
        # 召回效果评估
        metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)

