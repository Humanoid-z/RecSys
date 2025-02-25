# import numpy as np
# import pandas as pd
# import pickle
# from tqdm import tqdm
# import gc, os
# import logging
# import time
# import lightgbm as lgb
# from gensim.models import Word2Vec
# from sklearn.preprocessing import MinMaxScaler
# import warnings

# from utils import reduce_mem
# warnings.filterwarnings('ignore')

# data_path = '../data/'
# save_path = '../temp_results/'


# # 分割训练 验证集 sample_user_nums 采样作为验证集的用户数量
# def trn_val_split(all_click_df, sample_user_nums):
#     all_click = all_click_df
#     all_user_ids = all_click.user_id.unique()
    
#     # replace=True表示可以重复抽样，反之不可以
#     sample_user_ids = np.random.choice(all_user_ids, size=sample_user_nums, replace=False) 
    
#     click_val = all_click[all_click['user_id'].isin(sample_user_ids)]
#     click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]
    
#     # 将验证集中的最后一次点击给抽取出来作为答案
#     click_val = click_val.sort_values(['user_id', 'click_timestamp'])
#     val_ans = click_val.groupby('user_id').tail(1)
    
#     click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)
    
#     # 去除val_ans中某些用户只有一个点击数据的情况，如果该用户只有一个点击数据，又被分到ans中，
#     # 那么训练集中就没有这个用户的点击数据，出现用户冷启动问题，给自己模型验证带来麻烦
#     val_ans = val_ans[val_ans.user_id.isin(click_val.user_id.unique())] # 保证答案中出现的用户在验证集中还有
#     click_val = click_val[click_val.user_id.isin(val_ans.user_id.unique())]
    
#     return click_trn, click_val, val_ans
# # 获得训练 验证 测试集
# def get_trn_val_tst_data(data_path, offline=True,sample_user_nums=10000):
#     if offline:
#         click_trn_data = pd.read_csv(data_path+'train_click_log.csv')  # 训练集用户点击日志
#         click_trn_data = reduce_mem(click_trn_data)
#         click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums)
#     else:
#         click_trn = pd.read_csv(data_path+'train_click_log.csv')
#         click_trn = reduce_mem(click_trn)
#         click_val = None
#         val_ans = None
    
#     click_tst = pd.read_csv(data_path+'testA_click_log.csv')
    
#     return click_trn, click_val, click_tst, val_ans
# # 读取数据
# # 获取历史点击和最后一次点击做训练label
# def get_hist_and_last_click(all_click):
#     all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
#     click_last_df = all_click.groupby('user_id').tail(1)

#     # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
#     def hist_func(user_df):
#         if len(user_df) == 1:
#             return user_df
#         else:
#             return user_df[:-1]

#     click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

#     return click_hist_df, click_last_df

# click_trn, click_val, click_tst, val_ans = get_trn_val_tst_data(data_path, offline=True)
# click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn) ## 获取历史点击和最后一次点击做训练label
# print(click_trn_hist, click_trn_last)

# click_trn_hist, click_trn_last = get_hist_and_last_click(click_trn) # 验证集的label
# if click_val is not None:
#     click_val_hist, click_val_last = click_val, val_ans
# else:
#     click_val_hist, click_val_last = None, None
    
# click_tst_hist = click_tst  

import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    train_users = df_train_click['user_id'].values.tolist()
    # 随机采样出一部分样本
    val_users = sample(train_users, 50000)
    log.debug(f'val_users num: {len(set(val_users))}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = [] 

    groups = df_train_click.groupby(['user_id'])
    for user_id, g in tqdm(groups):
        if user_id in val_users:
            valid_query = g.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            click_list.append(g)

    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    test_query_list = []    # query 就是最后的点击id  label

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],   
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../data/train_click_log.csv')
    df_test_click = pd.read_csv('../data/testA_click_log.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)