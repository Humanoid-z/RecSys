import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from random import shuffle
import time
import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='hotcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default=f'test{time.time()}.log')
parser.add_argument('--beg_hours', default=0.7,type=float)
parser.add_argument('--end_hours', default=0.7,type=float)
args = parser.parse_args()

mode = args.mode
logfile = args.logfile
beg_hours = args.beg_hours
end_hours = args.end_hours

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'hotcf 召回，mode: {mode}')

# @multitasking.task
# def hot_recall(df_query,df_click, hot_articles, hot_articles_dict, worker_id):  #发布时间在点击时间范围内最热
#     last_click = df_click.drop_duplicates(subset='user_id', keep='last')[['user_id', 'click_timestamp', 'click_article_id']].reset_index(drop=True)
#     last_click_time = last_click.set_index('user_id')['click_timestamp'].to_dict()
    
#     user_item_ = df_click.groupby('user_id')['click_article_id'].agg( # 每个user点击item用list保存
#         lambda x: list(x)).reset_index()
#     user_item_dict = dict(
#         zip(user_item_['user_id'], user_item_['click_article_id']))
#     data_list = []
#     for user_id, item_id in tqdm(df_query.values):
#         topk_click = []
#         click_time = last_click_time[user_id]
#         for article_id in hot_articles['article_id'].unique():
#             if article_id in user_item_dict[user_id]:
#                 continue
#             min_time = click_time - 24 * 60 * 60 * 1000
#             max_time = click_time + 24 * 60 * 60 * 1000/8
#             if not min_time <= hot_articles_dict[article_id] <= max_time:
#                 continue
#             topk_click.append(article_id)
#             if len(topk_click) == 50:
#                 break
#         df_temp = pd.DataFrame()
#         df_temp['article_id'] = topk_click
#         df_temp['user_id'] = user_id
#         if  item_id == -1:  # online
#             df_temp['label'] = np.nan
#         else:
#             df_temp['label'] = 0 # 没命中
#             df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  #命中
#         df_temp = df_temp[['user_id', 'article_id', 'label']]
#         df_temp['user_id'] = df_temp['user_id'].astype('int')
#         df_temp['article_id'] = df_temp['article_id'].astype('int')
#         data_list.append(df_temp)
#     df_data = pd.concat(data_list, sort=False)

#     os.makedirs('../user_data/tmp/hotcf', exist_ok=True)
#     df_data.to_pickle(f'../user_data/tmp/hotcf/{worker_id}.pkl')
@multitasking.task
def hot_recall(df_query,df_click, articles_time, worker_id):  #发布时间在点击时间范围内最热
    last_click = df_click.drop_duplicates(subset='user_id', keep='last')[['user_id', 'click_timestamp', 'click_article_id']].reset_index(drop=True)
    last_click_time = last_click.set_index('user_id')['click_timestamp'].to_dict()
    
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg( # 每个user点击item用list保存
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    data_list = []
    for user_id, item_id in tqdm(df_query.values):
        topk_click = []
        click_time = last_click_time[user_id]
        min_time = click_time - beg_hours * 60 * 60 * 1000
        max_time = click_time + end_hours * 60 * 60 * 1000
        articles_time_temp = articles_time[(articles_time['click_timestamp'] >= min_time) & (articles_time['click_timestamp'] <= max_time)]

        article_counts = articles_time_temp['article_id'].value_counts()
        hot_articles = pd.DataFrame(article_counts.index.to_list(), columns=['article_id'])
        hot_articles['click_count'] = article_counts.values

        # hot_articles = hot_articles.merge(articles_time_temp[['article_id', 'click_timestamp']].drop_duplicates('article_id', keep='last'), on='article_id', how='left')
        # hot_articles = hot_articles.sort_values(by=['click_count','click_timestamp'], ascending=[False,False])
        # hitrate_50: 0.746211933031944
        # print(hot_articles) # Todo created_at_ts
        # exit()
 
        viewed_articles = user_item_dict[user_id]
        # 假设 viewed_articles 是一个包含已查看文章ID的列表或集合
        topk_click = hot_articles[~hot_articles['article_id'].isin(viewed_articles)][:50][['article_id','click_count']]
       
        df_temp = pd.DataFrame(topk_click)
        df_temp['user_id'] = user_id

        if  item_id == -1:  # online
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0 # 没命中
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  #命中
        df_temp = df_temp[['user_id', 'article_id','click_count','label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        df_temp['click_count'] = df_temp['click_count'].astype('int')
        data_list.append(df_temp)
    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/hotcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/hotcf/{worker_id}.pkl')

  


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        user_item_dict_file = '../user_data/sim/offline/user_item_dict.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)


    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    df_click = df_click.sort_values(['user_id', 'click_timestamp']) #升序    
    df_article = pd.read_csv('../data/articles.csv')

    articles_time = df_click[['click_article_id','click_timestamp']].sort_values(['click_article_id', 'click_timestamp'])
    articles_time.columns = ['article_id', 'click_timestamp']
    articles_time = articles_time.merge(df_article).drop(columns=['category_id', 'words_count'])    # 加上创建信息
    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/hotcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_click_temp = df_click[df_click['user_id'].isin(part_users)]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        hot_recall(df_temp, df_click_temp, articles_time, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/hotcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data,df_temp])

    # 对分数进行排序
    df_data = df_data.sort_values(['user_id','click_count'],
                                  ascending=[True,False]).reset_index(drop=True)
    df_data.columns = ['user_id', 'article_id','sim_score','label'] #统一columns
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'hotcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(f'beg_hours {beg_hours} end_hours {end_hours} hitrate_50: {hitrate_50}')
    # 保存召回结果
    df = df_data['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_hot.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_hot.pkl')



# beg_hours 8.0 end_hours 1.0 hitrate_50: 0.7113120373590343
# beg_hours 6.0 end_hours 1.0 hitrate_50: 0.7216205474688261
# beg_hours 4.0 end_hours 1.0 hitrate_50: 0.7325500521635452
# beg_hours 2.0 end_hours 1.0 hitrate_50: 0.7429330816235282
# beg_hours 1 end_hours 1 hitrate_50: 0.7462367728153411
# beg_hours 0.5 end_hours 1.0 hitrate_50: 0.7463361319489294
# beg_hours 0.7 end_hours 1.0 hitrate_50: 0.7465596899995032
# hotcf: 0.3621392021461573, 0.20338483448422906, 0.5008942322022952, 0.22186867916225958, 0.6238014804510905, 0.23051741461125955, 0.7196085250136619, 0.23395380737216476, 0.7465596899995032, 0.2345513757707427

# beg_hours 0.7 end_hours 0.8 hitrate_50: 0.7468826071836654
# beg_hours 0.7 end_hours 0.7 hitrate_50: 0.7472303641512246 ⭐
# beg_hours 0.7 end_hours 0.6 hitrate_50: 0.7469571265338566