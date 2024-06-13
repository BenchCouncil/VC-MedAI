from src.utils.utils_dataloader import *


root = '/home/ddcui/doctor/'
first_fn_pkl = root + f'datasets/sepsis_model_input/first_data_7000_dim_35.pkl'
first_diag_data = Data(fn=first_fn_pkl, flag='first')

final_fn_pkl = root + f'datasets/sepsis_model_input/final_data_7000_dim_36.pkl'
final_diag_data = Data(fn=final_fn_pkl, flag='final')

sepsis_patient_first_dict = {}
sepsis_patient_final_dict = {}

normal_0h_patient_first_dict = {}
normal_0h_patient_final_dict = {}

for patid, emb in zip(first_diag_data.patient_id, first_diag_data.embedding):
    if patid not in (list(sepsis_patient_first_dict.keys())):
        sepsis_patient_first_dict[patid] = emb[-17:]

for patid, emb in zip(final_diag_data.patient_id, final_diag_data.embedding):
    if patid not in (list(sepsis_patient_final_dict.keys())):
        emb = emb[:-1]
        sepsis_patient_final_dict[patid] = emb[-17:]


#专用模型：初步 最后17维，最终  减去最后一位后得到最后17维度
#普适模型

if __name__ == '__main__':
    print()

