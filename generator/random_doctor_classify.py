from src.data_process.embedding.doctor_embedding import doctor_unit_dict,doctor_title_dict,doctor_depart_dict


def abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag):
    test_uuid_new = []
    test_patient_id_new = []
    test_virdoc_diag_new = []

    for index, uuid in enumerate(test_uuid):
        if uuid in uuid_list:
            patid = test_patient_id[index]
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_virdoc_diag_new.append(virdoc_diag)
    combine_param = (test_uuid_new, test_patient_id_new,  test_virdoc_diag_new)
    return combine_param


#患者分类，脓毒症患者和非脓毒症患者
def patient_exper_3h_label1(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
    else:
        df = df_sample[df_sample['TIME_RANGE'].isnull()]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_virdoc_diag_new


def patient_exper_3h_label0(patient_flag,df_sample,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param

    if patient_flag == 'sepsis':
        df = df_sample[df_sample['ILL_TIME'].notnull()]
        df = df[df['TIME_RANGE'] != '3h']
    else:
        df = df_sample[(df_sample['TIME_RANGE'].isnull()) | (df_sample['TIME_RANGE'] == '3h')]

    sepsis_patientid = list(df['UNIQUE_ID'])
    test_uuid_new = []
    test_patient_id_new = []
    test_virdoc_diag_new = []

    for index,uuid in enumerate(test_uuid):
        patid = test_patient_id[index]
        if patid in sepsis_patientid:
            virdoc_diag = test_virdoc_diag[index]

            test_uuid_new.append(uuid)
            test_patient_id_new.append(patid)
            test_virdoc_diag_new.append(virdoc_diag)
    return test_uuid_new,test_patient_id_new,test_virdoc_diag_new


#模型分类
def model_exper(model_sort,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param
    if model_sort == '无模型':
        df = df_log[~df_log['model_sort'].notna()]
    else:
        df = df_log[df_log['model_sort'].notna()]
        df = df[df['model_sort'].str.contains(model_sort)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

#模型可见性分类
def model_visexper(model_sort,model_vis,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param
    df = df_log[df_log['model_sort'].notna()] #只统计有模型的情况下

    if model_sort == None:
        df = df[df['model_visible'] == model_vis]
    else:
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_visible'] == model_vis)]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)


#医生单位等级
def doctor_unit(unit,df_log,combine_param):
    if unit == '二甲':
        df = df_log[df_log['virdoc_unit'] == 1]
    elif unit == '三甲':
        df = df_log[df_log['virdoc_unit'] == 2]
    else:
        df = df_log[df_log['virdoc_unit'] == 3]
    test_uuid, test_patient_id, test_virdoc_diag = combine_param
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

def df_doctor_unit(df_log):
    df_se = df_log[df_log['virdoc_unit'] == 1]
    df_th = df_log[df_log['virdoc_unit'] == 2]
    df_u = df_log[df_log['virdoc_unit'] == 3]

    return df_se,df_th,df_u


#医生性别
def doctor_sex(sex,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param
    df = df_log[df_log['virdoc_sex'] == sex]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

#医生年龄
def doctor_age(minage,maxage,df_log,combine_param):
    test_uuid, test_patient_id,  test_virdoc_diag = combine_param
    df = df_log[(df_log['virdoc_age'] <= maxage) & (df_log['virdoc_age'] > minage)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

#医生从业年限
def doctor_year(min_year,max_year,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param

    df_log = df_log[df_log['virdoc_year'].notnull()]
    df = df_log[(df_log['virdoc_year'] <= max_year) & (df_log['virdoc_year'] > min_year)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

#医生职称
def doctor_title(title,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param

    df_log = df_log[df_log['virdoc_title'].notnull()]
    if title == '无':
        df = df_log[df_log['virdoc_title'] == doctor_title_dict.get('无')]
    elif title == '住院医生':
        df = df_log[(df_log['virdoc_title'] == doctor_title_dict.get('住院医师')) | (df_log['virdoc_title'] == doctor_title_dict.get('初级'))]
    elif title == '主治医生':
        df = df_log[
            (df_log['virdoc_title'] == doctor_title_dict.get('主治')) | (df_log['virdoc_title'] == doctor_title_dict.get('主治医师')) | (df_log['virdoc_title'] == doctor_title_dict.get('主治医生'))]
    else :
        #title == '副/主任医生':
        df = df_log[(df_log['virdoc_title'] == doctor_title_dict.get('副主任医师')) | (df_log['virdoc_title'] == doctor_title_dict.get('副主任医生')) | (
                    df_log['virdoc_title'] == doctor_title_dict.get('副主任'))| (df_log['virdoc_title'] == doctor_title_dict.get('主任医师'))]

    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list,test_uuid,test_patient_id,test_virdoc_diag)

def df_doctor_title(df_log):

    df_log = df_log[df_log['virdoc_title'].notnull()]
    df_none = df_log[df_log['virdoc_title'] == doctor_title_dict.get('无')]
    df_low = df_log[(df_log['virdoc_title'] == doctor_title_dict.get('住院医师')) | (
                df_log['virdoc_title'] == doctor_title_dict.get('初级'))]
    df_med = df_log[
        (df_log['virdoc_title'] == doctor_title_dict.get('主治')) | (
                    df_log['virdoc_title'] == doctor_title_dict.get('主治医师')) | (
                    df_log['virdoc_title'] == doctor_title_dict.get('主治医生'))]
    # title == '副/主任医生':
    df_high = df_log[(df_log['virdoc_title'] == doctor_title_dict.get('副主任医师')) | (
                df_log['virdoc_title'] == doctor_title_dict.get('副主任医生')) | (
                             df_log['virdoc_title'] == doctor_title_dict.get('副主任')) | (
                                 df_log['virdoc_title'] == doctor_title_dict.get('主任医师'))]

    return df_none,df_low,df_med,df_high


#医生科室
def doctor_depart(depart_list,df_log,combine_param):
    test_uuid, test_patient_id, test_virdoc_diag = combine_param
    depart_int_list = set()
    for depart in depart_list:
        depart_int = doctor_depart_dict.get(depart)
        depart_int_list.add(depart_int)
    df = df_log[df_log['virdoc_depart'].isin(depart_int_list)]
    uuid_list = list(df['uuid'])
    return abstract_idlist(uuid_list, test_uuid, test_patient_id, test_virdoc_diag)


#模型可见性分类
def df_model_visexper(model_sort,model_vis,df_log):
    df = df_log[df_log['model_sort'].notna()] #只统计有模型的情况下

    if model_sort == None:
        df = df[df['model_visible'] == model_vis]
    else:
        df = df[(df['model_sort'].str.contains(model_sort)) & (df['model_visible'] == model_vis)]
    print(f'模型是否显示：{model_vis},数量为：{len(df)}')
    return df


def docdiag_same_or_diff(df_randomdoc,randomdoc_uuids, randomdoc_diags):
    mapping_dict = {
        '脓毒症':1,
        '正常':0,
        '脓毒症预警':0
    }
    uuid_to_diag = {uuid: diag for uuid, diag in zip(randomdoc_uuids, randomdoc_diags)}
    df_randomdoc['virdoc_diag'] = df_randomdoc['uuid'].map(uuid_to_diag)

    df_randomdoc['model_pre'] = df_randomdoc['model_pre'].map(mapping_dict)

    df_diagsame = df_randomdoc[df_randomdoc['model_pre'] == df_randomdoc['virdoc_diag']]
    df_diagdiff = df_randomdoc[df_randomdoc['model_pre'] != df_randomdoc['virdoc_diag']]
    print(f'真实医生和模型诊断一致的数量：{len(df_diagsame)},诊断不一致的数量:{len(df_diagdiff)}')
    return df_diagsame,df_diagdiff
