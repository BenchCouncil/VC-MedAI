import pandas as pd


def abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag):
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index, uuid in enumerate(test_uuid):
        if uuid in uuid_list:
            patid = test_patient_id[index]
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    combine_param = (test_uuid_new, test_patient_id_new, test_truedoc_diag_new, test_virdoc_diag_new)
    return combine_param


#患者分类，脓毒症患者和非脓毒症患者
def patient_exper_3h_label0(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
        df = df[df['TIME_RANGE'] != '3h']
    else:
        df = df_sample[(df_sample['TIME_RANGE'].isnull()) | (df_sample['TIME_RANGE'] == '3h')]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_truedoc_diag_new,test_virdoc_diag_new


def patient_exper_3h_label1(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
    else:
        df = df_sample[df_sample['TIME_RANGE'].isnull()]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_truedoc_diag_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            truedoc_diag = test_truedoc_diag[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_truedoc_diag_new.append(truedoc_diag)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_truedoc_diag_new,test_virdoc_diag_new

#模型分类
def model_exper(model_sort,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    if model_sort == '无模型':
        df = df_log[~df_log['model_sort'].notna()]
    else:
        df = df_log[df_log['model_sort'].notna()]
        df = df[df['model_sort'].str.contains(model_sort)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def df_model_exper(model_sort,df_log):
    if model_sort == '无模型':
        df = df_log[~df_log['model_sort'].notna()]
    else:
        df = df_log[df_log['model_sort'].notna()]
        df = df[df['model_sort'].str.contains(model_sort)]
    return df

def df_model_sort_result(model_sort,model_pre,df_log):
    if model_sort == '无模型':
        df = df_log[~df_log['model_sort'].notna()]
    else:
        df = df_log[df_log['model_sort'].notna()]
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_pre']==model_pre)]
    return df


#模型可见性分类
def model_visexper(model_sort,model_vis,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['model_sort'].notna()] #只统计有模型的情况下

    if model_sort == None:
        df = df[df['model_visible'] == model_vis]
    else:
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_visible'] == model_vis)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def df_model_visexper(model_sort,model_vis,df_log):
    df = df_log[df_log['model_sort'].notna()] #只统计有模型的情况下

    if model_sort == None:
        df = df[df['model_visible'] == model_vis]
    else:
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_visible'] == model_vis)]

    return df

#医生单位等级
def doctor_unit(unit,df_log,combine_param):
    if unit == '二甲':
        df = df_log[df_log['doctor_unit'] == 1]
    elif unit == '三甲':
        df = df_log[df_log['doctor_unit'] == 2]
    else:
        df = df_log[df_log['doctor_unit'] == 3]
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)



#医生性别
def doctor_sex(sex,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['doctor_sex'] == sex]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)



#医生年龄
def doctor_age(minage,maxage,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[(df_log['doctor_age'] <= maxage) & (df_log['doctor_age'] > minage)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

#医生从业年限
def doctor_year(min_year,max_year,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    df_log = df_log[df_log['doctor_year'].notnull()]
    df = df_log[(df_log['doctor_year'] <= max_year) & (df_log['doctor_year'] > min_year)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

#医生职称
def doctor_title(title,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param

    df_log = df_log[df_log['doctor_title'].notnull()]
    if title == '无':
        df = df_log[df_log['doctor_title'] == '无']
    elif title == '住院医生':
        df = df_log[(df_log['doctor_title'] == '住院医师') | (df_log['doctor_title'] == '初级')]
    elif title == '主治医生':
        df = df_log[
            (df_log['doctor_title'] == '主治') | (df_log['doctor_title'] == '主治医师') | (df_log['doctor_title'] == '主治医生')]
    else :
        #title == '副/主任医生':
        df = df_log[(df_log['doctor_title'] == '副主任医师') | (df_log['doctor_title'] == '副主任医生') | (
                    df_log['doctor_title'] == '副主任')| (df_log['doctor_title'] == '主任医师')]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_truedoc_diag,test_virdoc_diag)

def df_doctor_title(df_log):

    df_log = df_log[df_log['doctor_title'].notnull()]
    df_none = df_log[df_log['doctor_title'] == '无']
    df_low = df_log[(df_log['doctor_title'] == '住院医师') | (df_log['doctor_title'] == '初级')]
    df_med = df_log[
        (df_log['doctor_title'] == '主治') | (df_log['doctor_title'] == '主治医师') | (df_log['doctor_title'] == '主治医生')]
    # title == '副/主任医生':
    df_high = df_log[(df_log['doctor_title'] == '副主任医师') | (df_log['doctor_title'] == '副主任医生') | (
            df_log['doctor_title'] == '副主任') | (df_log['doctor_title'] == '主任医师')]

    return df_none,df_low,df_med,df_high


#医生科室
def doctor_depart(depart_list,df_log,combine_param):
    test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag = combine_param
    df = df_log[df_log['doctor_depart'].isin(depart_list)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list, test_uuid, test_patient_id, test_truedoc_diag, test_virdoc_diag)


#患者分类，脓毒症患者和非脓毒症患者
def df_patient_exper_3h_nonsepsis(patient_flag,df_sample,df_truedoc):

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
        df = df[df['TIME_RANGE'] != '3h']
    else:
        df = df_sample[(df_sample['TIME_RANGE'].isnull()) | (df_sample['TIME_RANGE'] == '3h')]
    sepsis_patientid = list(df['UNIQUE_ID'])
    df = df_truedoc[(df_truedoc['UNIQUE_ID'].isin(sepsis_patientid))]
    return df

#在计算误诊率的时候 用的这个分脓毒症患者和非脓毒症患者
def df_patient_exper_3h_sepsis(patient_flag,df_sample,df_truedoc):

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
    else:
        df = df_sample[(df_sample['ILL_TIME'].isnull())]
    sepsis_patientid = list(df['UNIQUE_ID'])
    df = df_truedoc[(df_truedoc['UNIQUE_ID'].isin(sepsis_patientid))]
    return df



def docdiag_same_or_diff(df_truedoc):

    def convert_diag(row):
        label_dict = {
            '无脓毒症': 0,
            '低度疑似脓毒症': 1,
            '高度疑似脓毒症': 1,
            '一般脓毒症': 1,
            '严重脓毒症': 1
        }
        matched_value = 9999

        # 遍历字典的键
        for key in label_dict:
            if key in str(row['final_diag']):
                matched_value = label_dict[key]
                break
        return matched_value

    mapping_dict = {
        '脓毒症':1,
        '正常':0,
        '脓毒症预警':0
    }
    df_truedoc['final_diag'] = df_truedoc.apply(convert_diag, axis=1)
    df_truedoc['model_pre'] = df_truedoc['model_pre'].map(mapping_dict)

    df_diagsame = df_truedoc[df_truedoc['model_pre'] == df_truedoc['final_diag'].astype(int)]
    df_diagdiff = df_truedoc[df_truedoc['model_pre'] != df_truedoc['final_diag'].astype(int)]
    # print(f'真实医生和模型诊断一致的数量：{len(df_diagsame)},诊断不一致的数量:{len(df_diagdiff)}')
    return df_diagsame,df_diagdiff


def add_detail_range(df_sample):

    for index,row in df_sample.iterrows():
        admittime = row['ADMITTIME']
        illtime = row['ILL_TIME']
        if str(illtime) == 'nan':
            continue
        ill_starttime,ill_endtime = get_illtime_from_admittime(admittime,illtime)

        illtime_1 = ill_starttime - pd.Timedelta(hours=3)
        illtime_2 = ill_starttime - pd.Timedelta(hours=2)
        illtime_3 = ill_starttime - pd.Timedelta(hours=1)

        illtime_4 = ill_endtime + pd.Timedelta(hours=1)
        illtime_5 = ill_endtime + pd.Timedelta(hours=2)
        illtime_6 = ill_endtime + pd.Timedelta(hours=3)

        currenttime = row['START_ENDTIME']
        current_starttime = pd.to_datetime(currenttime.split('~')[0])
        current_timerange = row['TIME_RANGE']

        if current_timerange == '0h':
            df_sample.at[index,'time_range_detail'] = '0h'
        elif current_timerange == '3h':#患病前
            if current_starttime == illtime_1:
                df_sample.at[index, 'time_range_detail'] = '3h'
            elif current_starttime== illtime_2:
                df_sample.at[index, 'time_range_detail'] = '2h'
            elif current_starttime== illtime_3:
                df_sample.at[index, 'time_range_detail'] = '1h'

        elif current_timerange == '-3h':
            if current_starttime == ill_endtime:
                df_sample.at[index, 'time_range_detail'] = '-1h'
            elif current_starttime == illtime_4 :
                df_sample.at[index, 'time_range_detail'] = '-2h'
            elif current_starttime == illtime_5 :
                df_sample.at[index, 'time_range_detail'] = '-3h'

    return df_sample


def get_illtime_from_admittime(admittime,ill_time):
    ill_time = pd.to_datetime(ill_time)
    admittime = pd.to_datetime(admittime)
    for i in range(336):
        starttime = admittime + pd.Timedelta(hours=i+1)
        endtime = starttime+ pd.Timedelta(hours=1)
        if ill_time>= starttime and ill_time<= endtime:
            return starttime,endtime
    return None,None