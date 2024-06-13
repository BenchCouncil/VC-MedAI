import sys
from project_path import pro_path
sys.path.append(pro_path)

import pickle
import os
from simulator.data_process.embedding.model_embedding import *
from get_patemb_from_model_input import sepsis_patient_first_dict,sepsis_patient_final_dict



def get_randomdoc_nextact_by_uuid(uuid,uuid_list,nextact_list):
    if uuid in uuid_list:
        index = uuid_list.index(uuid)
        return nextact_list[index]
    return 0


def read_nextact_pkl(topath):
    global uuid_list, nextact_list
    loaded_data = []
    with open(topath, 'rb') as file:
        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass  # End of file reached
    uuid_list, nextact_list = zip(*loaded_data)
    return uuid_list, nextact_list

def first_embedding(df,dict_patient_embeddings,model_sort,topath,feature_numdim,clickseq):
    for ind, row in df.iterrows():
        patinet_id = row['UNIQUE_ID']
        uuid = row['uuid']
        model_emb = model_embedding(row, model_sort)
        doctor_emb = np.array([row['diag_order'],row['virdoc_unit'],row['virdoc_sex'],row['virdoc_age'],row['virdoc_year'],row['virdoc_title'],row['virdoc_field'],row['virdoc_depart']])

        if model_sort == 'sepsis':#保留患者信息
            patient_embedding = dict_patient_embeddings.get(patinet_id)
            if patient_embedding is None:
                print('-----没有患者信息---------------------')
                continue
            cat_embedding = np.concatenate((doctor_emb, model_emb, patient_embedding), axis=0) #np.array(patient_embedding).flatten()
            if len(cat_embedding) != feature_numdim:
                print('模型特征总数不对')
            else:
                data_to_save = (uuid, patinet_id,cat_embedding, None, None, None)
                with open(topath, 'ab') as file:
                    pickle.dump(data_to_save, file)
        else:#普适模型 不保留患者信息

            cat_embedding = np.concatenate((doctor_emb, model_emb), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('模型特征总数不对')
            data_to_save = (uuid, patinet_id, cat_embedding,None, None, None)
            with open(topath, 'ab') as file:
                pickle.dump(data_to_save, file)



def final_embedding(df, dict_patient_embeddings, model_sort, topath,feature_numdim,uuid_list, nextact_list):
    for ind, row in df.iterrows():
        patinet_id = row['UNIQUE_ID']
        uuid = row['uuid']
        model_emb = model_embedding(row, model_sort)
        doctor_emb = np.array([row['diag_order'],row['virdoc_unit'],row['virdoc_sex'],row['virdoc_age'],row['virdoc_year'],row['virdoc_title'],row['virdoc_field'],row['virdoc_depart']])
        predict_action = get_randomdoc_nextact_by_uuid(uuid, uuid_list, nextact_list)

        if model_sort == 'sepsis':  # 保留患者信息
            patient_embedding = dict_patient_embeddings.get(patinet_id)
            if patient_embedding is None:
                print('-----没有患者信息---------------------')
                continue
            cat_embedding = np.concatenate((doctor_emb, model_emb, patient_embedding, [predict_action]), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('模型特征总数不对')
            data_to_save = (uuid, patinet_id, cat_embedding, None, None)
            with open(topath, 'ab') as file:
                pickle.dump(data_to_save, file)
        else:  # 普适模型 不保留患者信息

            cat_embedding = np.concatenate((doctor_emb, model_emb, [predict_action]), axis=0)
            if len(cat_embedding) != feature_numdim:
                print('模型特征总数不对')
            data_to_save = (uuid, patinet_id, cat_embedding, None, None)
            with open(topath, 'ab') as file:
                pickle.dump(data_to_save, file)



root = '/home/ddcui/doctor/datasets/csv_and_pkl/'
def model_final_input_emb(model_sort,flag,feature_numdim,pca_dim,clickseq):

    to_root = f'/home/ddcui/doctor/datasets/randomdoc_model_input/'
    if not os.path.exists(to_root):
        os.makedirs(to_root)

    df = pd.read_csv(root + f'data_randomdoc.csv', encoding='gbk')

    # patient_embedding = root + f'patient_embedding_{flag}_randomdoc.pkl'
    # unique_id_list, patient_embedding_list = read_patient_emb(patient_embedding)
    # dim_embedding_list = pca(pca_dim, patient_embedding_list)
    # dictionary = dict(zip(unique_id_list, dim_embedding_list))
    if flag == 'first' and model_sort == 'sepsis':
        dictionary = sepsis_patient_first_dict
    elif flag == 'final' and model_sort == 'sepsis':
        dictionary = sepsis_patient_final_dict
    else:
        dictionary = None

    to_fn = to_root + f'{model_sort}_{flag}_data_dim_{feature_numdim}_randomdoc.pkl'
    if os.path.exists(to_fn):
        os.remove(to_fn)
    if flag == 'first':
        first_embedding(df, dictionary, model_sort, to_fn,feature_numdim,clickseq)
    else:

        uuid_list, nextact_list = read_nextact_pkl(to_root+f'{model_sort}_model_ramdom_doctor_nextact_predict.pkl')
        final_embedding(df, dictionary, model_sort, to_fn,feature_numdim,uuid_list, nextact_list)


if __name__ == '__main__':
    # 专用模型，初步还是最终(first,final)，最终模型的维度都加1 下一步特征
    # start_time = time.time()
    # model_final_input_emb('sepsis','first',35,17,False)  #第三个参数总维度（模型+患者+医生），第四个维度是患者信息降到多少维度 35=pat:17dim+ doc:8dim+model:10dim
    # end_time = time.time()
    # elapsed_minutes = (end_time - start_time) / 60
    # print(f"初步诊断的特征嵌入 程序运行时间: {elapsed_minutes:.2f} 分钟，也就是{(end_time - start_time)}秒")

    # start_time = time.time()
    # model_final_input_emb('sepsis','final',36,17,False) #最终模型的
    # end_time = time.time()
    # elapsed_minutes = (end_time - start_time) / 60
    # print(f"最终诊断的特征嵌入 程序运行时间: {elapsed_minutes:.2f} 分钟，也就是{(end_time - start_time)}秒")

    # model_final_input_emb('normal_0h', 'first', 17, None, False)
    # model_final_input_emb('normal_3h', 'first', 17, None, False)

    model_final_input_emb('normal_0h', 'final', 18, None, False)
    model_final_input_emb('normal_3h', 'final', 18, None, False)