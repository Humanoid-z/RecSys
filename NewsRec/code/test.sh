time=$(date "+%Y-%m-%d-%H:%M:%S")
# 处理数据
# python preprocess.py --mode valid --logfile "${time 处理数据}.log"

# # itemcf 召回
# python recall_itemcf.py --mode valid --logfile "${time itemcf 召回}.log"

# # binetwork 召回
# python recall_binetwork.py --mode valid --logfile "${time binetwork 召回}.log"

# # w2v 召回
# python recall_w2v.py --mode valid --logfile "${time w2v 召回}.log"

# # 召回合并
# python recall.py --mode valid --logfile "${time 召回合并}.log"

# # 排序特征
# python rank_feature.py --mode valid --logfile "${time 排序特征}.log"

# # lgb 模型训练
# python rank_lgb.py --mode valid --logfile "${time lgb 模型训练}.log"




# python recall_hot.py --mode valid --beg_hours 0.7 --end_hours 0.9

# python recall_hot.py --mode valid --beg_hours 0.7 --end_hours 0.7

# python recall_hot.py --mode valid --beg_hours 0.7 --end_hours 0.6


# 召回合并
python recall.py --mode valid --binetwork_w 0.8 --w2v_w 0.1 --hot_w 1
python recall.py --mode valid --binetwork_w 0.8 --w2v_w 0.1 --hot_w 2
python recall.py --mode valid --binetwork_w 0.8 --w2v_w 0.1 --hot_w 3
python recall.py --mode valid --binetwork_w 0.8 --w2v_w 0.1 --hot_w 0.8
python recall.py --mode valid --binetwork_w 0.8 --w2v_w 0.1 --hot_w 0.6
