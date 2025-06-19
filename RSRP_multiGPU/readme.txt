experiment里面是主实验，所有文件都一样只是任务/网络不同，多卡版本。
里面可以调的参数包括:
随机种子;网络尺寸;采样数量，pop_size是总数量，subpop_num是分多少次串行，device_num是分多张卡并行；奖励函数；优化器；
reward_transform，奖励函数的regularization；兴奋抑制网络，是否double input/兴奋抑制神经元比例；weight_transform，参数矩阵的regularization；
输出在wandb，fitness是训练集的reward，evaluation是验证集(测试集的一个batch)acc，test是测试集acc，entropy是参数矩阵的信息熵。

ec.modules：有MLP和CNN等网络结构；
ec.evo_state/core/optim：梯度更新函数；
ec.evo_config：超参数和选择；
ec.metrics：计算reward和信息熵；
ec.ops：继承的代码，应该是加速作用，但没有用到；
3个dataloader，原版（每次sampling都随机抽一个数据）、1（按顺序抽batch抽完一个epoch，好像慢一点点）和2（随机抽batch）