import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         vector_size=256,   #256
                         window=5,  #3
                         min_count=1,
                         sg=1,  # 1
                         hs=0,  #0
                         seed=seed,
                         negative=15,# 5
                         workers=10,
                         epochs=3)  #1
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model.wv.key_to_index:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map

@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        interacted_items = user_item_dict[user_id][0]
        interacted_items = interacted_items[::-1][:1]     # 取最后一位
        interacted_items_time = user_item_dict[user_id][1]
        interacted_items_time = interacted_items_time[::-1][:1]
        interacted_items_time = [np.abs(interacted_items_time[0]-time)/(1000*3600*10) for time in interacted_items_time]
        for loc,item in enumerate(interacted_items):
            article_vec = article_vec_map[item]

            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    w1 =  (0.3**interacted_items_time[loc])   # 越早点击的interacted_items的relate_item权重越小
                    w2 =  (0.7**loc)
                    rank[relate_item] += wij*w1*w2

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))

# @multitasking.task
# def recall(df_query, article_vec_map, article_index, user_item_dict,
#            worker_id):
#     data_list = []

#     for user_id, item_id in tqdm(df_query.values):
#         rank = defaultdict(int)

#         interacted_items = user_item_dict[user_id]
#         interacted_items = interacted_items[-1:]    # 取最后一位

#         for loc,item in enumerate(interacted_items):
#             article_vec = article_vec_map[item]

#             item_ids, distances = article_index.get_nns_by_vector(
#                 article_vec, 100, include_distances=True)
#             sim_scores = [2 - distance for distance in distances]

#             for relate_item, wij in zip(item_ids, sim_scores):
#                 if relate_item not in interacted_items:
#                     rank.setdefault(relate_item, 0)
#                     rank[relate_item] += wij

#         sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
#         item_ids = [item[0] for item in sim_items]
#         item_sim_scores = [item[1] for item in sim_items]

#         df_temp = pd.DataFrame()
#         df_temp['article_id'] = item_ids
#         df_temp['sim_score'] = item_sim_scores
#         df_temp['user_id'] = user_id

#         if item_id == -1:
#             df_temp['label'] = np.nan
#         else:
#             df_temp['label'] = 0
#             df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

#         df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
#         df_temp['user_id'] = df_temp['user_id'].astype('int')
#         df_temp['article_id'] = df_temp['article_id'].astype('int')

#         data_list.append(df_temp)

#     df_data = pd.concat(data_list, sort=False)

#     os.makedirs('../user_data/tmp/w2v', exist_ok=True)
#     df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    pickle.dump(article_vec_map, f)
    f.close()

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    # user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
    #     lambda x: list(x)).reset_index()
    # user_item_dict = dict(
    #     zip(user_item_['user_id'], user_item_['click_article_id']))
    user_item_ = df_click.groupby('user_id')['click_article_id','click_timestamp'].agg( # 每个user点击item用list保存
        lambda x: list(x)).reset_index()
    # print(user_item_)
    user_item_dict = dict(
        zip(user_item_['user_id'], list(zip(user_item_['click_article_id'],user_item_['click_timestamp']))))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')

# 取最后5位  w1 底数 
# 0.3064235679864872  0.5
# 0.32594763773659896 0.4
# 0.3777634259029261  0.3
# 0.3459188235878583  0.2
# w2 底数  0.7
# 0.30783943564012123 0.6
# 0.3196134929703413  0.8



# 0.39251825724079686 epoch 2
# 0.3925927765909881 3 √
# 0.36340603109940883 4




# 0.34035471210691043

# 0.30754135823935613 wondow 6
# 0.368498186695812   5 √
# 0.30157981022405483 4

# negtive 5
# 0.39254309702419393 15 √
# 0.2841422822792985  20
# 0.30557901535098614 10

# 取最后n位  1
# 0.17435043966416613 5