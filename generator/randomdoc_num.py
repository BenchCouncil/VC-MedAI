import pandas as pd
from src.data_process.embedding.doctor_embedding import *
import random
import time


#增加随机种子
random.seed(123)

def random_value_by_prob(df, column_name):
    column_values = df[column_name]
    value_counts = column_values.value_counts()
    total_count = len(column_values)
    probabilities = value_counts / total_count
    value_prob_dict = probabilities.to_dict()
    values = list(value_prob_dict.keys())
    probabilities = list(value_prob_dict.values())
    chosen_value = random.choices(values, probabilities)[0]
    return chosen_value


def generate_doctor(num,df):
    random_doc_list = []
    title_counts = df['doctor_title'].value_counts()
    title_proba = title_counts / len(title_counts)
    title_prob_dict = title_proba.to_dict()

    title_num_dict = {}
    total_weight = sum(title_prob_dict.values())
    for key, prob in title_prob_dict.items():
        weight = prob / total_weight
        title_num_dict[key] = round(weight * num)

    for title in title_num_dict.keys():
        num_kind = title_num_dict.get(title)
        # print(f'医生职级为 {title}，人数为{num_kind}')
        for i in range(num_kind):
            df_t = df[df['doctor_title'] == title]
            group_unit = random_value_by_prob(df_t, 'doctor_unit')
            group_sex = random_value_by_prob(df_t, 'doctor_sex')
            group_age = random_value_by_prob(df_t, 'doctor_age')
            group_year = random_value_by_prob(df_t, 'doctor_year')
            group_depart = random_value_by_prob(df_t, 'doctor_depart')
            group_field = random_value_by_prob(df_t, 'doctor_field')
            random_doc_list.append((group_unit, group_sex, group_age, group_year, title, group_field, group_depart))
    return random_doc_list


path = '/home/ddcui/doctor/datasets/csv_and_pkl/data_0321_7000.csv'
df = pd.read_csv(path,encoding='gbk',usecols=['uuid','diag_seq','doctor_name','doctor_unit','doctor_sex','doctor_age','doctor_year','doctor_depart','doctor_title','doctor_field'])

df['doctor_unit'] = df['doctor_unit'].map(doctor_unit_dict)
df['doctor_sex'] = df['doctor_sex'].map(doctor_sex_dict)
df['doctor_depart'] = df['doctor_depart'].map(doctor_depart_dict)
df['doctor_title'] = df['doctor_title'].map(doctor_title_dict)
df['doctor_field'] = df['doctor_field'].map(doctor_field_dict)

random_doc_list = generate_doctor(125,df)




def time_of_genrandomdoc():
    #统计生成虚拟医生的时间
    start_time = time.time()

    generate_doctor(125, df)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"程序运行时间: {elapsed_minutes:.2f} 分钟，也就是{elapsed_time}秒")


if __name__ == '__main__':
    time_of_genrandomdoc()



