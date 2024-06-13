
root = '/home/ddcui/doctor/'
final_sample_data = root + f'datasets/initial_data/样例数据_20231120.csv'
log_data = root + f'datasets/csv_and_pkl/data_0321_7000.csv'
randomdoc_model_input_path = root + 'datasets/randomdoc_model_input/'
sepsis_3h_label = 'sepsis_1'  # sepsis_1, nonsepsis_0

# ---------------专用模型--------------
model_sort = 'sepsis'
model_input_path = root + 'datasets/sepsis_model_input/'
model_path =root +  f'model_save/sepsis_model/'
first_diag_modelname = 'first_diag_dim_35_kford_2_sen_0.75_spe_0.78_acc_77.0_auc_83.25.dat'
click_sequence_modelname = ''
final_diag_modelname = 'final_diag_dim_36_kford_1_sen_0.73_spe_0.8_acc_78.06_auc_84.79.dat'

first_diagtime_modelname = 'first_diagtime_dim_35_kford_1_mae_0.4242.dat'
final_diagtime_modelname = 'final_diagtime_dim_36_kford_1_mae_0.759.dat'
feature_dimnum_first = 35
feature_dimnum_final = 36
first_or_final = 'final'  #final,first
ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_nextact_predict.pkl'
ramdoc_diag_file = root + f'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_{first_or_final}_diag.pkl'
ramdoc_diagtime_file = root + f'datasets/randomdoc_model_input/sepsis_model_ramdom_doctor_{first_or_final}_diagtime.pkl'


# # ---------------普适模型0h--------------
# model_sort = 'normal_0h'
# model_input_path = root + 'datasets/normal_0h_model_input/'
# model_path =root +  f'model_save/normal_0h_model/'
# first_diag_modelname = 'first_diag_dim_17_kford_1_sen_0.7_spe_0.77_acc_74.32_auc_80.02.dat'
# first_nextact_modelname = 'first_nextact_dim_17_kford_1_auc_86.31_acc_70.2.dat'
# final_diag_modelname = 'final_diag_dim_18_kford_4_sen_0.75_spe_0.8_acc_78.52_auc_83.56.dat'
#
# first_diagtime_modelname = 'first_diagtime_dim_17_kford_1_mae_0.4618.dat'
# final_diagtime_modelname = 'final_diagtime_dim_18_kford_1_mae_0.7768.dat'
# feature_dimnum_first = 17
# feature_dimnum_final = 18
# first_or_final = 'final'  # first final
# ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/normal_0h_model_ramdom_doctor_nextact_predict.pkl'
# ramdoc_diag_file = None
# ramdoc_diagtime_file = None



# # ---------------普适模型3h--------------
# model_sort = 'normal_3h'
# model_input_path = root + 'datasets/normal_3h_model_input/'
# model_path =root +  f'model_save/normal_3h_model/'
# first_diag_modelname = 'first_diag_dim_17_kford_4_sen_0.77_spe_0.74_acc_75.23_auc_80.53.dat'
# first_nextact_modelname = 'first_nextact_dim_17_kford_1_auc_87.02_acc_68.53.dat'
# final_diag_modelname = 'final_diag_dim_18_kford_4_sen_0.75_spe_0.8_acc_78.67_auc_83.42.dat'
#
# first_diagtime_modelname = 'first_diagtime_dim_17_kford_1_mae_0.459.dat'
# final_diagtime_modelname = 'final_diagtime_dim_18_kford_1_mae_0.7756.dat'
# feature_dimnum_first = 17
# feature_dimnum_final = 18
# first_or_final = 'first'  #final first
# ramdoc_nextact_file = root + 'datasets/randomdoc_model_input/normal_3h_model_ramdom_doctor_nextact_predict.pkl'
# ramdoc_diag_file = None





