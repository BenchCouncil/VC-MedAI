from torch.utils.data import Dataset
import random
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from simulator.utils.utils_io_pkl import *

class Data(Dataset):
    def __init__(self, fn, flag):
        click_seq_list = None
        percent_action_list = None
        diag_list = None
        diag_time_list = None
        seq_list = None
        if 'clickseq' in fn:
            uuid_list,patientid_list,embedding_list, click_seq_list = read_click_emb(fn)
            # uuid_list, patientid_list, seq_list, embedding_list, click_seq_list = read_click_emb_patdiff(fn)

        elif flag == 'first':
            uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list = read_first_emb(fn)
        else:
            uuid_list,patientid_list,embedding_list, diag_list, diag_time_list,percent_action_list = read_final_emb(fn)

        embedding_list = np.array(embedding_list)
        embedding_list = np.where(embedding_list is None, 0.0, embedding_list)
        embedding_list = embedding_list.astype(float)
        embedding_list = np.nan_to_num(embedding_list, nan=0.0, posinf=0.0, neginf=0.0)

        self.uuid = uuid_list
        self.patient_id = patientid_list
        self.embedding = embedding_list
        self.diag = diag_list
        self.diag_time = diag_time_list
        self.percent_action = percent_action_list
        self.click_seq = click_seq_list
        self.seq = seq_list


    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        return self.embedding[index], self.diag[index], self.diag_time[index]

    def get_emb_by_uuid(self,uuid):
        index = self.uuid.index(uuid)
        # 返回对应的 embedding
        return self.embedding[index]

    def get_diag_by_uuid(self,uuid):
        index = self.uuid.index(uuid)
        # 返回对应的 embedding
        return self.diag[index]

    def get_clickseq_by_uuid(self,uuid):
        uuids = list(self.uuid)
        index = uuids.index(uuid)
        # 返回对应的 embedding
        click_seqs = list(self.click_seq)
        return click_seqs[index]
    # 去除一个患者多个医生诊断的情况
    def remove_multi_doctor_diag(self):
        unique_patient_ids = set()
        unique_indices = []

        # 找出唯一的patient_id及其索引
        for i, pid in enumerate(self.patient_id):
            if pid not in unique_patient_ids:
                unique_patient_ids.add(pid)
                unique_indices.append(i)

        # 使用找到的索引更新列表
        self.uuid = [self.uuid[i] for i in unique_indices]
        self.patient_id = [self.patient_id[i] for i in unique_indices]
        self.embedding = [self.embedding[i] for i in unique_indices]
        self.diag = [self.diag[i] for i in unique_indices]
        self.diag_time = [self.diag_time[i] for i in unique_indices]
        # self.percent_action = [self.percent_action[i] for i in unique_indices]




def feature_scaling(data):
    # 创建MinMaxScaler实例，设置范围为0到10
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 应用缩放器到数据
    scaled_data = scaler.fit_transform(np.array(data.embedding))
    data.embedding = scaled_data
    return data

def to_xgb(train_set):
    embedding = np.array(train_set.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.array(train_set.diag)
    diag_time = np.array(train_set.diag_time)
    percent_action = np.array(train_set.percent_action)
    return embedding, diag, diag_time,percent_action,

def randomdoc_to_xgb(data1):
    embedding = np.array(data1.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    data1.embedding = embedding
    return data1


def to_click_seq(train_set):
    embedding = np.array(train_set.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    click_seq = np.array(train_set.click_seq)

    return embedding,click_seq


def to_xgb_data(train_set):
    embedding = np.array(train_set.embedding)
    embedding = np.where(np.array(embedding) is None, 0.0, embedding)
    embedding = embedding.astype(float)
    train_set.embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    return train_set


def balance_data(data):
    random.seed(5)

    # 统计每个类别的数量
    class_counts = {label: data.diag.count(label) for label in set(data.diag)}
    print(class_counts)

    # 找到最多的类别和对应数量
    max_class, max_count = max(class_counts.items(), key=lambda x: x[1])

    # 对于每个类别，将其样本数量补齐到最多的数量
    augmented_embedding = []
    augmented_diag = []
    augmented_diag_time = []

    for label in set(data.diag):
        # 计算需要复制的数量
        copies_needed = max_count - class_counts[label]

        # 获取当前类别的索引列表
        indices = [i for i, value in enumerate(data.diag) if value == label]

        # 随机复制样本
        for _ in range(copies_needed):
            random_index = random.choice(indices)
            augmented_embedding.append(data.embedding[random_index])
            augmented_diag.append(data.diag[random_index])
            augmented_diag_time.append(data.diag_time[random_index])

    # 将原始数据和复制的数据合并
    balanced_embedding = list(data.embedding) + augmented_embedding
    balanced_diag = list(data.diag) + augmented_diag
    balanced_diag_time = list(data.diag_time) + augmented_diag_time

    # 更新类别数量
    balanced_class_counts = {label: balanced_diag.count(label) for label in set(balanced_diag)}
    print(balanced_class_counts)

    data.embedding = tuple(balanced_embedding)
    data.diag = tuple(balanced_diag)
    data.diag_time = tuple(balanced_diag_time)
    return data


def stratify_sampling_diag(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag, test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp,random_state=10)
    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.diag = y_train
    val_set.embedding = X_val
    val_set.diag = y_val
    test_set.embedding = X_test
    test_set.diag = y_test

    train_emb, train_diag, _,_ = to_xgb(train_set)
    val_emb, val_diag, _,_ = to_xgb(val_set)
    test_emb, test_diag, _,_ = to_xgb(test_set)
    return train_emb,train_diag, val_emb,val_diag, test_emb,test_diag




def sampling_diag_of_uuid(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.diag, test_size=0.3,random_state=42)
    X_val, test_uuid, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp,random_state=10)

    X_train1, X_temp1, y_train1, y_temp1 = train_test_split(data.patient_id, data.diag, test_size=0.3,
                                                        random_state=42)
    X_val, test_patient_id, y_val, _ = train_test_split(X_temp1, y_temp1, test_size=0.32, stratify=y_temp1, random_state=10)

    return test_uuid,test_patient_id,y_test

def sampling_diag_of_uuid_val(data):
    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp, random_state=10)

    X_train_uuid, X_temp, y_train_uuid, y_temp = train_test_split(data.uuid, data.diag, test_size=0.3, random_state=42)
    X_val_uuid, X_test, y_val_uuid, y_test = train_test_split(X_temp, y_temp, test_size=0.32, stratify=y_temp, random_state=10)

    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    kflod_uuid = np.concatenate((X_train_uuid,X_val_uuid))
    kflod_em = np.concatenate((X_train,X_val))
    kflod_label = np.concatenate((y_train,y_val))
    val_uuid = None
    val_emb = None
    val_diag = None
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        _, val_uuid = kflod_uuid[train_index], kflod_uuid[val_index]
        _, val_emb = kflod_em[train_index], kflod_em[val_index]
        _, val_diag = kflod_label[train_index], kflod_label[val_index]
        break #最终诊断的kfold是1
    return val_uuid,val_emb,val_diag


def stratify_sampling_diagtime(data,flag):
    data = select_data(data,flag)

    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.diag_time, test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42)

    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.diag_time = y_train
    val_set.embedding = X_val
    val_set.diag_time = y_val
    test_set.embedding = X_test
    test_set.diag_time = y_test
    train_emb, _, train_diagtime, _ = to_xgb(train_set)
    val_emb, _, val_diagtime, _ = to_xgb(val_set)
    test_emb, _, test_diagtime, _ = to_xgb(test_set)
    return train_emb,train_diagtime, val_emb,val_diagtime, test_emb,test_diagtime

def sampling_diagtime_of_uuid(data,flag):
    data = select_data(data, flag)

    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.diag_time, test_size=0.3, random_state=42)
    X_val, X_uuid, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32, random_state=42)

    X_train, X_temp, y_train, y_temp = train_test_split(data.patient_id, data.diag_time, test_size=0.3, random_state=42)
    X_val, X_patient_id, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.32, random_state=42)
    return X_uuid,X_patient_id,y_test

def select_data(data,flag):
    embedding_list = []
    diag_time_list = []
    uuid_list =[]
    patid_list =[]

    if flag == 'first': #均值是2.5
        max_value = 7.5 #放大3倍
        min_value = 0.5 #均值的20%
    else:#均值是3.8
        max_value = 15.2 #放大4倍
        min_value = 0.76 #均值的20%
    for i,time in enumerate(data.diag_time):
        if time > max_value or time < min_value:
            print('诊断质量可能不高 去掉')
        else:
            uuid_list.append(data.uuid[i])
            patid_list.append(data.patient_id[i])
            embedding_list.append(data.embedding[i])
            diag_time_list.append(data.diag_time[i])
    data.embedding =embedding_list
    data.diag_time = diag_time_list
    data.uuid = uuid_list
    data.patient_id = patid_list
    return data


def stratify_sampling_nextact(data):
    #全部查看就2条数据，去掉，改成8分类了
    data.uuid = [uuid for uuid, percent_action in zip(data.uuid, data.percent_action) if percent_action != 1]
    data.patient_id = [patient_id for patient_id, percent_action in zip(data.patient_id, data.percent_action) if
                       percent_action != 1]
    data.embedding = [embedding for embedding, percent_action in zip(data.embedding, data.percent_action) if
                      percent_action != 1]
    data.diag = [diag for diag, percent_action in zip(data.diag, data.percent_action) if percent_action != 1]
    data.diag_time = [diag_time for diag_time, percent_action in zip(data.diag_time, data.percent_action) if
                      percent_action != 1]
    data.percent_action = [percent_action for percent_action in data.percent_action if percent_action != 1]

    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)

    train_set = copy.deepcopy(data)
    val_set = copy.deepcopy(data)
    test_set = copy.deepcopy(data)

    train_set.embedding = X_train
    train_set.percent_action = y_train
    val_set.embedding = X_val
    val_set.percent_action = y_val
    test_set.embedding = X_test
    test_set.percent_action = y_test
    train_emb, _, _, train_next_act = to_xgb(train_set)
    val_emb, _, _, val_next_act = to_xgb(val_set)
    test_emb, _, _, test_next_act = to_xgb(test_set)
    return train_emb,train_next_act, val_emb,val_next_act, test_emb,test_next_act

def sampling_nextact_of_uuid(data):
    data.uuid = [uuid for uuid, percent_action in zip(data.uuid, data.percent_action) if percent_action != 1]
    data.patient_id = [patient_id for patient_id, percent_action in zip(data.patient_id, data.percent_action) if
                       percent_action != 1]
    data.embedding = [embedding for embedding, percent_action in zip(data.embedding, data.percent_action) if
                      percent_action != 1]
    data.percent_action = [percent_action for percent_action in data.percent_action if percent_action != 1]
    X_train, X_temp, y_train, y_temp = train_test_split(data.uuid, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)
    
    X_train, X_temp, y_train, y_temp = train_test_split(data.patient_id, data.percent_action, test_size=0.3,random_state=42,stratify=data.percent_action)

    X_val, x_patid, y_val,_ = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42,stratify=y_temp)
 
    return X_test,x_patid,y_test

def stratify_sampling_clickseq(data,num_class_num,end_class):
    def each_reducedim(emb):
        emb = np.array(emb)
        ranges = [(0, 8, 4), (8, 18, 6), (18, 35, 10)] #医生信息降维到2，模型信息降维到3，患者信息降维到5
        reduced_data_list = []
        for start, end, n_components in ranges:
            subset_data = emb[:, start:end]
            pca = PCA(n_components=n_components)
            reduced_subset_data = pca.fit_transform(subset_data)
            reduced_data_list.append(reduced_subset_data)
        reduced_data = np.hstack(reduced_data_list)
        return reduced_data

    data.embedding = each_reducedim(data.embedding)
    X_train, X_temp, y_train, y_temp = train_test_split(data.embedding,data.click_seq, test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.32,random_state=42)

    maxlength = 0
    for click_seq in data.click_seq:
        if len(click_seq) > maxlength:
            maxlength = len(click_seq)
    print(f'医生最长点击序列为 {maxlength}')

    def xgb_clickseq(sample_startid,X_train,y_train):
        embedding_after_list = []
        seq_class_list = []
        sampleid_list = []
        clickseqid_list = []
        for embedding,seqlist in zip(X_train,y_train):
            #把uuid和序列id添加到embedding中
            seq_id = 1
            if num_class_num == 11: #初步诊断之前的序列建模
                new_seqlist = [x for x in seqlist if x < 10]
            elif num_class_num == 7: #下一步检查的序列建模
                new_seqlist = [x - 10 for x in seqlist if (x >= 10) and (x<16)] #因为分类器必须是从0开始  所以统一减去10，预测的时候加10就行
            else:
                new_seqlist = seqlist

            new_seqlist.append(end_class)  # 添加终止符类别 10基础操作的结束 8下一步检查的结束符  18
            for seq_class in new_seqlist:
                embedding_after = np.concatenate(([seq_id], embedding))
                embedding_after_list.append(embedding_after)
                seq_class_list.append(seq_class)
                sampleid_list.append(sample_startid)
                clickseqid_list.append(seq_id)
                seq_id += 1
            sample_startid +=1
        return sampleid_list,clickseqid_list,embedding_after_list,seq_class_list

    X_trainid_list,X_train_seqid_list,X_train_after,y_train_after = xgb_clickseq(1,X_train,y_train)
    X_valid_list,X_val_seqid_list,X_val_after,y_val_after = xgb_clickseq(5001,X_val,y_val)
    X_testid_list,X_test_seqid_list,X_test_after,y_test_after = xgb_clickseq(10001, X_test,y_test)
    print(f'total_data {len(data.click_seq)}')
    print(f'训练集样本数 {len(set(X_trainid_list))}')
    print(f'验证集样本数 {len(set(X_valid_list))}')
    print(f'测试集样本数 {len(set(X_testid_list))}')

    return X_train_after,y_train_after, X_val_after,y_val_after,X_testid_list,X_test_seqid_list,X_test_after,y_test_after

def stratify_sampling_clickseq_patdiff(data):
    uuid_list = list(set(data.uuid))
    total_length = len(uuid_list)
    part1_length = int(total_length * 0.7)
    part2_length = int(total_length * 0.2)

    train_uuid = uuid_list[:part1_length]
    val_uuid = uuid_list[part1_length:part1_length + part2_length]
    test_uuid = uuid_list[part1_length + part2_length:]
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for uuid,emb,click_class in zip(data.uuid,data.embedding,data.click_seq):
        if uuid in train_uuid:
            X_train.append(emb)
            y_train.append(click_class)
        elif uuid in val_uuid:
            X_val.append(emb)
            y_val.append(click_class)
        else:
            X_test.append(emb)
            y_test.append(click_class)
    print(f'total_data {len(uuid_list)}')
    print(f'训练集样本数 {len(train_uuid)}')
    print(f'验证集样本数 {len(val_uuid)}')
    print(f'测试集样本数 {len(test_uuid)}')

    return X_train,y_train, X_val,y_val,X_test,y_test



def multi_balance_data(X_train, y_train):
    # 统计每个类别的样本数量
    class_counts = Counter(y_train)

    # 找出样本数量最多的类别
    mean_samples = int(np.mean(list(class_counts.values())))

    # 初始化新数据集
    new_X_train = []
    new_y_train = []

    # 对于每个类别，进行过采样或欠采样
    for class_label, count in class_counts.items():
        if count > mean_samples:
            # 如果当前类别的样本数量超过最大样本数量，随机选择一部分样本进行删除
            class_indices = np.where(np.array(y_train) == class_label)[0]
            selected_indices = np.random.choice(class_indices, mean_samples, replace=False)
            new_X_train.extend(np.array(X_train)[selected_indices])
            new_y_train.extend(np.array(y_train)[selected_indices])
        else:
            # 如果当前类别的样本数量不足最大样本数量，进行过采样
            oversample_count = mean_samples - count
            class_indices = np.where(np.array(y_train) == class_label)[0]
            oversampled_indices = np.random.choice(class_indices, oversample_count, replace=True)
            new_X_train.extend(np.array(X_train)[oversampled_indices])
            new_y_train.extend(np.array(y_train)[oversampled_indices])
    # 将新数据集转换为数组格式
    new_X_train = np.array(new_X_train)
    new_y_train = np.array(new_y_train)

    return new_X_train, new_y_train


def lstm_fill_data(embedding_list, click_seq_list,timesteps):

    def clicklist_to_lstm_label(timesteps,click_seq_list, column_dim=18):
        click_seq_label = np.zeros((len(click_seq_list) ,column_dim))
        for i, val in enumerate(click_seq_list):
            click_seq_label[i, int(val)] = 1
        fill_label = np.full((timesteps-len(click_seq_list), column_dim), -4)
        concat_label = np.vstack((click_seq_label, fill_label))
        return concat_label

    def embedding_to_lstm_input(timesteps,embedding,click_len):
        embedding = embedding.astype(float)
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        embedding_after = np.empty((0, len(embedding)+1))
        for index in range(click_len):
            embedding_1 = np.append(embedding, index + 1)
            embedding_after = np.vstack((embedding_after, embedding_1))

        fill_array = np.full((timesteps - click_len, len(embedding)+1), -4)
        concat_embedding = np.vstack((embedding_after, fill_array))
        return concat_embedding

    embedding_list_after = []
    click_seqlist_after = []
    for (embedding, click_seq) in zip(embedding_list, click_seq_list):
        embedding_after = embedding_to_lstm_input(timesteps,embedding,len(click_seq))#添加了点击的顺序索引
        click_seq_label = clicklist_to_lstm_label(timesteps,click_seq)
        embedding_list_after.append(embedding_after)
        click_seqlist_after.append(np.array(click_seq_label))

    print(np.array(embedding_list_after).shape)
    print(np.array(click_seqlist_after).shape)

    return np.array(embedding_list_after,dtype=np.float32), np.array(click_seqlist_after,dtype=np.float32)



def data_split(data,flag, model):
    # if flag == 'final' :
    #     #初步模型采用所有数据 维度25，最终模型移出重复诊断 维度45
    #     Data.remove_multi_doctor_diag(data)

    data = feature_scaling(data)
    if model == 'diag':
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_diag(data)

    elif model == 'diagtime':
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_diagtime(data,flag)
    else :
        #simulator =='nextact'
        train_em, train_label, val_em, val_label, test_em, test_label = stratify_sampling_nextact(data)


    print(f'totaldata sum {len(train_em)+len(val_em)+len(test_em)}')
    if model != 'clickseq' and model != 'diagtime':
        print(f'train sum {len(train_em)}, 正样本数量 {sum(train_label)}, 负样本数量 {len(train_label) - sum(train_label)}')
        print(f'val sum {len(val_em)}, 正样本数量 {sum(val_label)}, 负样本数量 {len(val_label) - sum(val_label)}')
        print(f'test sum {len(test_em)}, 正样本数量 {sum(test_label)}, 负样本数量 {len(test_label) - sum(test_label)}')
    return np.concatenate((train_em,val_em)), np.concatenate((train_label,val_label)), test_em, test_label



if __name__ == '__main__':
    print()
