import pandas as pd
from scipy.stats import norm
import numpy as np
import re
import random


def convert_to_minutes(input_str):
    # 定义正则表达式模式，匹配天、小时和分钟
    input_str = input_str.replace('约','')
    pattern = re.compile(r'(?:(\d+)\s*天)?(?:(\d+)\s*小时)?(?:(\d+)\s*分钟)?')
    # 使用正则表达式匹配输入字符串
    match = pattern.match(input_str)
    # 提取匹配到的天、小时和分钟数值
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0
    # 将天、小时和分钟转换成总分钟数
    total_minutes = days * 24 * 60 + hours * 60 + minutes
    return total_minutes

def get_gaussiandis(df):
    time_min_list = []
    for index,row in df.iterrows():
        time = row['检查所用时间（单位 ）']
        time_min = convert_to_minutes(time)
        time_min_list.append(time_min)

    mean, std = norm.fit(time_min_list)
    mean = round(mean,2)
    std = round(std,2)
    print(f'mean {mean},std {std}')
    return mean,std

#预测每项检查的检查时间,时间的单位为分钟
def get_percheck_time(exam_type,subtype):
    time = None
    if exam_type == '降钙素原':
        time = np.random.normal(211.12, 326.98, 1)
    elif exam_type == '血常规':
        time = np.random.normal(31.1, 27.32, 1)
    elif exam_type == '动脉血气分析':
        time = np.random.normal(53.29, 66.82, 1)
    elif exam_type == '止凝血':
        time = np.random.normal(90.1, 65.17, 1)
    elif exam_type == '影像检查':
        if subtype == 'CT':
            time = np.random.normal(58.9, 71.98, 1)
        elif subtype == '核磁':
            time = np.random.normal(158.11, 165.45, 1)
        elif subtype == '超声':
            time = np.random.normal(21.62, 17.94, 1)
        else: # x光
            time = np.random.normal(25.5, 14.5, 1)
    elif exam_type == '病原检查':
        if subtype == '肺部相关':
            time = np.random.normal(199.15, 352.16, 1)
        elif subtype == '肝脏相关':
            time = np.random.normal(857.95, 1582.6, 1)
        elif subtype == '腺体相关':
            time = np.random.normal(477.87, 609.18, 1)
        else:  # 心血管系统相关
            time = np.random.normal(1027.75, 1349.61, 1)
    elif exam_type == '培养':
        if subtype in '伤口培养':
            time = np.random.normal(3023.71, 474.64, 1)
        elif subtype in '呼吸道培养|呼吸道病毒培养':
            time = np.random.normal(4826.86, 1867.79, 1)
        elif subtype in '尿液培养':
            time = np.random.normal(3832.67, 903.08, 1)
        elif subtype in '快速呼吸道病毒抗原测试':
            time = np.random.normal(317.33, 33.56, 1)
        elif subtype in '液体培养|液体培养在瓶中':
            time = np.random.normal(5193.67, 3718.55, 1)
        elif subtype in '真菌培养':
            time = np.random.normal(3165.5, 167.5, 1)
        elif subtype in '粪便培养 - 排除耶尔森氏菌|粪便培养':
            time = np.random.normal(4000.0, 1034.22, 1)
        elif subtype in '细菌培养':
            time = np.random.normal(4154.5, 238.5, 1)
        elif subtype in '血液培养，常规|血液/抗酸杆菌培养':
            time = np.random.normal(3773.75, 926.91, 1)
        else:  # 其他培养按照所有培养的高斯分布
            time = np.random.normal(3722.04, 1757.84, 1)
    elif exam_type == '涂片':
        #涂片大约 几个小时
        while True:
            time = np.random.normal(3722.04, 1757.84, 1)
            if time < 720:
                break
    if time is None:
        print(exam_type)
        print(subtype)
    while time < 0:
        time = get_percheck_time(exam_type,subtype)
    return time

#预测查看的数量为哪几项检查
def get_check_name(count):
    options = ['降钙素原', '血常规', '动脉血气分析', '止凝血', '影像检查', '病原检查', '培养', '涂片']
    sort_probability = [0.3364, 0.2476, 0.1832, 0.0513, 0.1185, 0.0269, 0.0276, 0.0085]
    doctor_choice = np.random.choice(options, size=count, replace=False, p=sort_probability)
    #key为主类型、value是子类型
    check_name_dict = {}
    for check in doctor_choice:
        if check == '影像检查':
            image_option = ['CT','胸部（前后位和侧面位）','胸部（便携式前后位）','核磁','超声','胸部前后位','胸部（单视图）','胸部卧位检查','胸部造影管置管 2 例检查','胸部造影管置管 1 例检查']
            image_probability = [0.3198,0.0571,0.5483,0.0053,0.0541,0.0122,0.0008,0.0008,0.0008,0.0008]
            sub_sort = np.random.choice(image_option, size=1, replace=False, p=image_probability)
            check_name_dict[check]=sub_sort[0]
        elif check == '病原检查':
            path_option = ['肺部相关','腺体相关','肝脏相关','心血管系统相关']
            path_probability = [0.8394,0.0067,0.0268,0.1271]
            sub_sort = np.random.choice(path_option, size=1, replace=False, p=path_probability)
            check_name_dict[check] = sub_sort[0]
        elif check == '培养':
            cul_option = ['呼吸道培养','尿液培养','厌氧瓶革兰氏染色','血液培养，常规','粪便培养 - 排除耶尔森氏菌','粪便培养','伤口培养','血液/抗酸杆菌培养','液体培养','病毒培养：排除单纯疱疹病毒','呼吸道病毒培养','液体培养在瓶中','真菌培养','快速呼吸道病毒抗原测试','抗酸杆菌（AFB）培养和涂片检查']
            cul_probability = [0.3639,0.0327,0.0722,0.3181,0.0098,0.0164,0.0229,0.0132,0.0196,0.0032,0.0263,0.0033,0.0885,0.0033,0.0066]
            sub_sort = np.random.choice(cul_option, size=1, replace=False, p=cul_probability)
            check_name_dict[check] = sub_sort[0]
        else:
            check_name_dict[check] = None
    return check_name_dict

def main_mean_std():
    root = 'D:\\4-work\\14-mimic-iv\\9-系统日志\\doctor\\datasets\\check_time\\'
    df1 = pd.read_csv(root + '1-降钙素原.csv')
    df2 = pd.read_csv(root + '2-血常规.csv')
    df3 = pd.read_csv(root + '3-动脉血气分析.csv')
    df4 = pd.read_csv(root + '4-止凝血.csv')
    df5 = pd.read_csv(root + '5-影像报告.csv')
    df6 = pd.read_csv(root + '6-病原检查.csv')
    df7 = pd.read_csv(root + '7-培养涂片.csv')

    print(f'-------------降钙素原------------')
    get_gaussiandis(df1)
    print(f'--------------血常规-------------')
    get_gaussiandis(df2)
    print(f'-------------动脉血气分析-----------')
    get_gaussiandis(df3)
    print(f'--------------止凝血---------------')
    get_gaussiandis(df4)

    print(f'--------------影像类型------------')
    grouped_df5 = df5.groupby('影像类型')
    for group_name, group_data in grouped_df5:
        print(f'{group_name},')
        get_gaussiandis(group_data)

    print(f'---------------病原检查--------------')
    grouped_df6 = df6.groupby('病原类型')
    for group_name, group_data in grouped_df6:
        print(f'{group_name},')
        get_gaussiandis(group_data)

    print(f'---------------培养涂片---------------')
    grouped_df7 = df7.groupby('培养/涂片类型')
    for group_name, group_data in grouped_df7:
        print(f'{group_name},')
        get_gaussiandis(group_data)

    get_gaussiandis(df7)


def get_totaltime_nextact(check_name_dict):
    #①使用模型预测下一步检查查看的数量
    #②预测为哪几项
    time_num = 0
    # 下面这个只要在
    # check_name_dict = get_check_name(next_viewcount)
    for key, value in check_name_dict.items():
        time = get_percheck_time(key,value)
        time_num = time_num + time
    return int(time_num) #时间单位为分钟

def minutes_to_days_hours_minutes(minutes):
    # 计算天数
    days = minutes // (24 * 60)
    # 计算剩余小时数
    hours = (minutes % (24 * 60)) // 60
    # 计算剩余分钟数
    remaining_minutes = minutes % 60
    return days, hours, remaining_minutes


if __name__ == '__main__':
    check_name_dict = get_check_name(3)
    totaltime = get_totaltime_nextact(check_name_dict)
    days, hours, remaining_minutes = minutes_to_days_hours_minutes(totaltime)
    print(check_name_dict)
    print(f'检查时间总计{totaltime}分钟，也就是 {days} 天 {hours} 小时 {remaining_minutes} 分钟')
