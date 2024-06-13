from project_path import *


#------模型参数---------
d_model = 256   # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8



batchsize = 64 #when training is 64
lr= 0.0001
epochs = 10
model_path = f'{pro_path}model_save/sepsis_model/transformer/'
test_model_name = 'model_set_acc_0.81_truedoc_seqlen_5.2_predict_seqlen_5.77.pth'


device = 'cuda:0'
tgt_vocab_size = 21     # 目标字典尺寸
src_vocab_size = 12239  # 字典字的个数
tgt_len = 21+2  #label的最大长度

train_resample = 10 #when training is 10
test_resample = 1
