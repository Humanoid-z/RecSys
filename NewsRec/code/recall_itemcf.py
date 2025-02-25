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
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default=f'test{time.time()}.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


# def cal_sim(df):
#     user_item_ = df.groupby('user_id')['click_article_id'].agg( # 每个user点击item用list保存
#         lambda x: list(x)).reset_index()
#     user_item_dict = dict(
#         zip(user_item_['user_id'], user_item_['click_article_id']))

#     item_cnt = defaultdict(int)
#     sim_dict = {}

#     for _, items in tqdm(user_item_dict.items()):
#         for loc1, item in enumerate(items):
#             item_cnt[item] += 1
#             sim_dict.setdefault(item, {})

#             for loc2, relate_item in enumerate(items):
#                 if item == relate_item: #找和item共现的其它relate_item
#                     continue

#                 sim_dict[item].setdefault(relate_item, 0)

#                 # 位置信息权重
#                 # 考虑文章的正向顺序点击和反向顺序点击
#                 loc_alpha = 1.0 if loc2 > loc1 else 0.5 # 用item找relate_item更好 0.7
#                 loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))   #距离约远关系越弱   Todo 根据现实时间改

#                 sim_dict[item][relate_item] += loc_weight  / \
#                     math.log(1 + len(items))        # 用用户点击长度加权，避免系统过度适配重度用户影响普通用户的结果

#     for item, relate_items in tqdm(sim_dict.items()):   
#         for relate_item, cij in relate_items.items():   #根据热门item加权相似度 避免热门item和所有都很相似
#             sim_dict[item][relate_item] = cij / \
#                 math.sqrt(item_cnt[item] * item_cnt[relate_item])

#     return sim_dict, user_item_dict
def cal_sim(df): #根据真实时间计算相似度
    user_item_ = df.groupby('user_id')['click_article_id','click_timestamp'].agg( # 每个user点击item用list保存
        lambda x: list(x)).reset_index()
    # print(user_item_)
    user_item_dict = dict(
        zip(user_item_['user_id'], list(zip(user_item_['click_article_id'],user_item_['click_timestamp']))))
    # print(list(user_item_dict.items())[:10])
    item_cnt = defaultdict(int)
    sim_dict = {}

    for _, items in tqdm(user_item_dict.items()):
        click_timestamps = items[1]
        items = items[0]
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item: #找和item共现的其它relate_item
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.5 # 用item找relate_item更好 0.7
                # loc_weight1 = loc_alpha * (0.9**(np.abs(click_timestamps[loc2] - click_timestamps[loc1])/(3600000*2)))   #距离约远关系越弱   Todo 根据现实时间改
                # loc_weight2 = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))
                # alpha = 0.6
                # loc_weight = alpha*loc_weight1+(1-alpha)*loc_weight2
                loc_weight1 = (0.9**(np.abs(click_timestamps[loc2] - click_timestamps[loc1])/(3600000*2)))   #距离约远关系越弱   Todo 根据现实时间改
                loc_weight2 = (0.9**(np.abs(loc2 - loc1) - 1))
                loc_weight = loc_alpha*loc_weight1*loc_weight2
                sim_dict[item][relate_item] += loc_weight  / \
                    math.log(1 + len(items))        # 用用户点击长度加权，避免系统过度适配重度用户影响普通用户的结果

    for item, relate_items in tqdm(sim_dict.items()):   
        for relate_item, cij in relate_items.items():   #根据热门item加权相似度 避免热门item和所有都很相似
            sim_dict[item][relate_item] = cij / \
                math.sqrt(item_cnt[item] * item_cnt[relate_item])

    # user_item_dict = dict(
    #     zip(user_item_['user_id'], user_item_['click_article_id']))
    return sim_dict, user_item_dict

@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:   # itemCF没有历史点击无法推荐相似item
            continue

        interacted_items = user_item_dict[user_id][0]
        interacted_items = interacted_items[::-1][:16]   # 限定只用最近点击的5个新闻来做召回 Todo 可以调
        interacted_items_time = user_item_dict[user_id][1]
        interacted_items_time = interacted_items_time[::-1][:16]
        interacted_items_time = [np.abs(interacted_items_time[0]-time)/(1000*3600*10) for time in interacted_items_time]
        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(item_sim[item].items(), 
                                           key=lambda d: d[1],  
                                           reverse=True)[0:400]:  #扩大相似item候选 前200->400个
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # rank[relate_item] += wij * (0.4**loc)   # 越早点击的interacted_items的relate_item权重越小 Todo 可以调 0.7
                    
                    # w1 = wij * (0.4**interacted_items_time[loc])   # 越早点击的interacted_items的relate_item权重越小
                    # w2 = wij * (0.4**loc)
                    # loc_alpha = 0.8
                    # rank[relate_item] += loc_alpha * w1 + (1-loc_alpha)*w2
                    w1 =  (0.5**interacted_items_time[loc])   # 越早点击的interacted_items的relate_item权重越小
                    w2 =  (0.75**loc)
                    rank[relate_item] += wij * w1*w2


        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]  # 前100召回
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0 # 没命中
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  #命中

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')
# @multitasking.task
# def recall(df_query, item_sim, user_item_dict, worker_id):
#     data_list = []

#     for user_id, item_id in tqdm(df_query.values):
#         rank = {}

#         if user_id not in user_item_dict:   # itemCF没有历史点击无法推荐相似item
#             continue

#         interacted_items = user_item_dict[user_id]
#         interacted_items = interacted_items[::-1][:5]   # 限定只用最近点击的5个新闻来做召回 Todo 可以调

#         for loc, item in enumerate(interacted_items):
#             for relate_item, wij in sorted(item_sim[item].items(), 
#                                            key=lambda d: d[1],  
#                                            reverse=True)[0:400]:  #扩大相似item候选 前200->400个
#                 if relate_item not in interacted_items:
#                     rank.setdefault(relate_item, 0)
#                     rank[relate_item] += wij * (0.4**loc)   # 越早点击的interacted_items的relate_item权重越小 Todo 可以调 0.7

#         sim_items = sorted(rank.items(), key=lambda d: d[1],
#                            reverse=True)[:100]  # 前100召回
#         item_ids = [item[0] for item in sim_items]
#         item_sim_scores = [item[1] for item in sim_items]

#         df_temp = pd.DataFrame()
#         df_temp['article_id'] = item_ids
#         df_temp['sim_score'] = item_sim_scores
#         df_temp['user_id'] = user_id

#         if item_id == -1:
#             df_temp['label'] = np.nan
#         else:
#             df_temp['label'] = 0 # 没命中
#             df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1  #命中

#         df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
#         df_temp['user_id'] = df_temp['user_id'].astype('int')
#         df_temp['article_id'] = df_temp['article_id'].astype('int')

#         data_list.append(df_temp)

#     df_data = pd.concat(data_list, sort=False)

#     os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
#     df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
        user_item_dict_file = '../user_data/sim/offline/user_item_dict.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    # log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()
    # f = open(user_item_dict_file, 'wb')
    # pickle.dump(user_item_dict, f)
    # f.close()
    # item_sim, user_item_dict = pickle.load(open(sim_pkl_file,'rb')),pickle.load(open(user_item_dict_file,'rb'))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data,df_temp])
            # df_data = df_data.append(df_temp)

    # 对分数进行排序
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
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(f'hitrate_50: {hitrate_50}')
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')


# 只用最近点击的n个新闻来做召回 old=n
# hitrate_50: 0.7285508470366139 6
# hitrate_50: 0.7287992448705848 8
# hitrate_50: 0.7289979631377614 16 √
# w2底数 0.7  最近点击的16个新闻来做召回 数量变多需要削弱loc的衰减
# hitrate_50: 0.7293705598887178 0.8
# hitrate_50: 0.7288489244373789 0.9
# hitrate_50: 0.7294947588057032 0.75 √ ★
# itemcf: 0.3588106711709474, 0.21430523125837542, 0.48477321277758456, 0.23103860062911954, 0.6015947140940932, 0.2391724132752781, 0.7019474390183318, 0.24275790278296594, 0.7294947588057032, 0.2433695128816499
# w1底数 0.5
# hitrate_50: 0.7287992448705848 0.6
# hitrate_50: 0.7293208803219235 0.4
# hitrate_50: 0.7292463609717323 0.45
# -------------------------------------------------------------------------------
# hitrate_50: 0.7093248546872671 base
# 召回用时间步衰减
# 衰减指数用n个小时 
# hitrate_50: 0.7192607680461026 1
# hitrate_50: 0.7184162154106016 0.5
# hitrate_50: 0.720229519598589  2
# hitrate_50: 0.7208256744001192 3
# hitrate_50: 0.7215211883352377 4
# hitrate_50: 0.7223160614039446 6
# hitrate_50: 0.7234338516568135 8
# hitrate_50: 0.7235332107904019 9
# hitrate_50: 0.7233096527398281 12
# hitrate_50: 0.7235332107904019
# hitrate_50: 0.7235828903571961 10 √

# 底数 0.4
# hitrate_50: 0.7122062695613294 0.8
# hitrate_50: 0.7216205474688261 0.6
# hitrate_50: 0.7231357742560485 0.3
# hitrate_50: 0.7231606140394455 0.5

# merge alpha
# hitrate_50: 0.72244026032093 0.5
# hitrate_50: 0.7245764816930796 0.8 √
# hitrate_50: 0.7244026032093001 0.9
# hitrate_50: 0.7245764816930796 0.7
# hitrate_50: 0.7237816086243728 0.6
# hitrate_50: 0.7243280838591087 0.75

# 尝试修改sim 指数n个时间 2 ×
# hitrate_50: 0.7223905807541359 4
# hitrate_50: 0.7245019623428883 1

# 尝试修改sim衰减系数的相乘融合 
# hitrate_50: 0.7270852998161856 √

# sim 衰减 loc_weight1底数 0.9 ×
# hitrate_50: 0.7239306473247553 0.8
# 尝试修改sim衰减系数的相乘融合+线性内插×
# hitrate_50: 0.7260420289135079

# 尝试修改recall衰减系数的相乘融合 
# hitrate_50: 0.7213473098514581
# w1底数 0.4
# hitrate_50: 0.7214218292016493 0.5
# hitrate_50: 0.7211982711510756 0.8
# hitrate_50: 0.7217199066024145 0.6
# w2底数 0.4
# hitrate_50: 0.7255949128123603 0.5
# hitrate_50: 0.7271598191663768 0.6
# hitrate_50: 0.7281534105022605 0.7 √
# hitrate_50: 0.7273585374335536 0.8
# w1底数 0.6
# hitrate_50: 0.7268120621988177 0.7
# hitrate_50: 0.72832728898604   0.5 √ ★
# itemcf: 0.35888519052113865, 0.21426217563382055, 0.48390382035868645, 0.2308772504646644, 0.6005017636246212, 0.23899755282576612, 0.7010283670326395, 0.24258420244969214, 0.72832728898604, 0.24318895787787084
# hitrate_50: 0.7282030900690546 0.4
# hitrate_50: 0.7281782502856575 0.45

# 尝试修改recall衰减系数的相乘融合+线性内插 alpha ×
# hitrate_50: 0.7265388245814497 0.8
# hitrate_50: 0.7274578965671419 0.9
# alpha1 1
# hitrate_50: 0.7268865815490089 0.5
# hitrate_50: 0.7278304933180982 1.5
# hitrate_50: 0.7282030900690546
# -------------------------------------------------------------------------
# 相似度
# hitrate_50: 0.6958368523026479 base
# 调整逆序权重
# hitrate_50: 0.6936757911471012 0.7->0.8
# hitrate_50: 0.6970540016891053 0.7->0.6
# hitrate_50: 0.697252719956282  0.7->0.5 √
# hitrate_50: 0.6956629738188683 0.7->0.4

# 换成时间戳按小时加权
# hitrate_50: 0.7044314173580406 1
# hitrate_50: 0.7004073724477122 0.5小时
# hitrate_50: 0.7069650752645437 2小时 √
# hitrate_50: 0.7068408763475582 4小时

# 时间加权 底数 0.9
# hitrate_50: 0.7012767648666104 0.7
# hitrate_50: 0.7041085001738785 0.8
# hitrate_50: 0.7068657161309554 0.95
# hitrate_50: 0.7054498484773213 0.85

# 时间戳和序列差融合  loc_alpha
# hitrate_50: 0.7085299816185603 0.5
# hitrate_50: 0.7093000149038701 0.7
# hitrate_50: 0.7093248546872671 0.6 √
# DEBUG: itemcf: 0.3512096974514382, 0.21137330882473876, 0.4712355308261712, 0.2273849740601708, 0.5830393958964678, 0.2351872054939522, 0.682150131650852, 0.23874320374238045, 0.7093248546872671, 0.2393442792122985

# --------------------------------------------------------------------
# 召回
# DEBUG: itemcf: 0.3333498931889314, 0.19656010399588225, 0.4541457598489741, 0.2126601968887956, 0.5663470614536241, 0.22049522182993542, 0.6589000943911769, 0.22382260576224527, 0.6849818669581201, 0.22440124590895272 base
# DEBUG: itemcf: 0.3331263351383576, 0.19639367744712183, 0.4542202791991654, 0.21253835380841, 0.5675145312732873, 0.2204336768835517, 0.6602414426946197, 0.2237542819622175, 0.6863232152615629, 0.22433229759200848 recall 相似item候选 200->300 relate_item 有一定提升
# DEBUG: itemcf: 0.3333747329723285, 0.19643135111860727, 0.45379800288141486, 0.21246524401735598, 0.5644840776988425, 0.22018849313121944, 0.6567390332356302, 0.2234998234568648, 0.6800635898454965, 0.22401722971320948 recall 相似item候选 200->100 relate_item 权重 下降
# DEBUG: itemcf: 0.33347409210591683, 0.19644542699586576, 0.45384768244820906, 0.21248936423559892, 0.5675145312732873, 0.22041958482001076, 0.6610114759799295, 0.2237655216558428, 0.6870435689800785, 0.2243414727930287 recall 相似item候选 200->400 relate_item 有一定提升 √

# recall 旧点击衰减更大
# DEBUG: itemcf: 0.33347409210591683, 0.19644542699586576, 0.45384768244820906, 0.21248936423559892, 0.5675145312732873, 0.22041958482001076, 0.6610114759799295, 0.2237655216558428, 0.6870435689800785, 0.2243414727930287 recall 相似item候选 200->400 relate_item 有一定提升 √
# DEBUG: itemcf: 0.33844204878533457, 0.2007981850398151, 0.45918823587858315, 0.21689567566970752, 0.5703214267971584, 0.22465957884512464, 0.6627005812509315, 0.22798166167331388, 0.6892543097024194, 0.2285711663602733  0.7->0.6
# hitrate_50: 0.6903969397386854  0.7->0.5
# hitrate_50: 0.6918624869591137 0.7->0.4 √
# hitrate_50: 0.6910427741070098 0.7->0.3

# 只用最近点击的n个新闻来做召回 old=2
# hitrate_50: 0.6958368523026479 2->5  √
# hitrate_50: 0.6957126533856625  2->7
# hitrate_50: 0.6958368523026479 2->10

# 尝试只用最近点击的10个新闻扩大衰减系数 ×
# hitrate_50: 0.6885091162005067 0.4 ->0.6
# hitrate_50: 0.6937503104972924 0.4 ->0.5

# 尝试只用最近点击的5个新闻扩大衰减系数  ×
# hitrate_50: 0.6937503104972924 0.4 ->0.5
# hitrate_50: 0.695290377067912 0.4 ->0.45

# 尝试只用最近点击的5个新闻减小相似item候选  ×
# hitrate_50: 0.6951910179343236 400->300
# ---------------------------------------------
# 召回最优参数 400 0.4 5