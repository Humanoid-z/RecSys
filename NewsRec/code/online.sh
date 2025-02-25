time=$(date "+%Y-%m-%d-%H:%M:%S")
# 处理数据
python preprocess.py --mode online --logfile "${time} 处理数据.log"
# hot 召回
python recall_hot.py --mode online --logfile "${time} hot 召回.log"
# itemcf 召回
python recall_itemcf.py --mode online --logfile "${time} itemcf 召回.log"

# binetwork 召回
python recall_binetwork.py --mode online --logfile "${time} binetwork 召回.log"

# w2v 召回
python recall_w2v.py --mode online --logfile "${time} w2v 召回.log"

# 召回合并
python recall.py --mode online --logfile "${time}召回合并.log"

# 排序特征
python rank_feature.py --mode online --logfile "${time}排序特征.log"

# lgb 模型训练
python rank_lgb.py --mode online --logfile "${time}lgb 模型训练.log"



