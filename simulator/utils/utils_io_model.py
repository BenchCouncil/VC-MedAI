# -*- encoding:utf -*-
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import pickle
import re
import os
from sklearn import metrics


def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model


def xgb_save_model(gs, model_path):
    path = model_path[:model_path.rfind('/')]
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(gs, open(model_path, "wb"))


# 加载模型
def xgb_load_model(model_path):
    loaded_model = pickle.load(open(model_path, "rb"))
    return loaded_model


def best_auc_model(folder_path, flag, model):
    if not os.path.exists(folder_path):
        return 0,0,None
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    best_auc = 0
    best_filname = ''
    for file_name in file_names:
        if flag in file_name and model in file_name:
            auc = extract_eval(file_name)
            if auc is not None and auc > best_auc:
                best_auc = auc
                best_filname = file_name
    if best_auc == 0:
        return None,best_auc, None
    if 'kford' in best_filname:
        match = re.search("kford_([0-9]+)", best_filname)
        best_kflod = match.group(1)
    else:
        best_kflod = None
    best_model = xgb_load_model(os.path.join(folder_path, best_filname))
    return best_kflod,best_auc, best_model


def lowest_rmse_model(folder_path, flag, model):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    lowest_rmse = 1000
    best_filname = ''
    for file_name in file_names:
        if flag in file_name and model in file_name:
            rmse = extract_eval(file_name)
            if rmse is not None and rmse < lowest_rmse:
                lowest_rmse = rmse
                best_filname = file_name
    if lowest_rmse == 1000:
        return None,lowest_rmse, None
    match = re.search("kford_([0-9]+)", best_filname)
    best_kflod = match.group(1)
    best_model = xgb_load_model(os.path.join(folder_path, best_filname))
    return best_kflod, lowest_rmse, best_model

#模型的评价指标都是最后一个值
def extract_eval(text):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if len(numbers) > 0:
        return float(numbers[-1])
    else:
        return None



def diag_eval(test_diag, y_pred, y_pred_prob):
    accuracy = round(accuracy_score(test_diag, y_pred), 4)
    auc = round(roc_auc_score(test_diag, y_pred_prob), 4)

    print(f'acc {accuracy},auc {auc}')
    return round(accuracy * 100, 2), round(auc * 100, 2)

def get_diff_senspe(test_diag, y_pred):
    tn, fp, fn, tp = confusion_matrix(test_diag, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    diff = abs(sensitivity-specificity)
    return round(diff,2),sensitivity,specificity


def rmse_eval(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values,predicted_values)
    print(f'Label的均值:{round(np.mean(true_values), 4)}，Label的标准差{round(np.std(true_values), 4)}，MAE: {round(mae, 4)}，RMSE: {round(rmse, 4)}')
    return round(rmse, 4),round(mae, 4)

#由于label和预测值直接算mae的时候，mae的大小是均值的36%左右，效果不好
#调整方法：将label转成范围，（label-label*20%）~（label+label*20%），选择两端中和预测值最近的时间
def diagtime_label_to_range(true_values, predicted_values,threshold):
    label_values = []

    for true, predicted in zip(true_values, predicted_values):
        low_value = max(true - true * threshold,0)
        high_value = true + true * threshold

        low_diff = abs(low_value - predicted)
        high_diff = abs(high_value - predicted)

        if low_diff < high_diff:
            label_value = low_value
        else:
            label_value = high_value
        label_values.append(label_value)
    return label_values

def multu_class_eval(true_values,predicted_values):
    y_pred = np.argmax(predicted_values, axis=1)
    acc = metrics.accuracy_score(true_values, y_pred)
    matrix = metrics.confusion_matrix(true_values, y_pred)
    auc_ovo = roc_auc_score(true_values, predicted_values, multi_class='ovo')
    auc_ovr = roc_auc_score(true_values, predicted_values, multi_class='ovr')

    print(f'acc: {round(acc,4)} , auc_ovo: {round(auc_ovo,4)} , auc_ovr: {round(auc_ovr,4)}')
    # print(f'混淆矩阵如下：')
    # print(matrix)
    # class_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    # # 计算每个类别的准确率
    # num_classes = len(matrix)
    # class_accuracies = {}
    # for i in range(num_classes):
    #     true_positive = matrix[i, i]
    #     false_positive = sum(matrix[:, i]) - true_positive
    #     class_accuracy = true_positive / (true_positive + false_positive)
    #     class_accuracies[class_names[i]] = class_accuracy
    #
    # print("Class Accuracies:")
    # for class_name, accuracy in class_accuracies.items():
    #     print(f"{class_name}: {accuracy}")
    return round(acc*100,2),round(auc_ovo*100,2)



def feature_important(model):
    feature_importance = model.feature_importances_

    # 打印每个特征的重要性
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i + 1}: {importance}")

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance, tick_label=f"Feature {i + 1}")
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.show()


def get_modelparams(xgb_model):
    xgb_params = xgb_model.get_params()

    print("Current XGBoost simulator parameters:")
    for param_name, param_value in xgb_params.items():
        print(f"{param_name}: {param_value}")


def chickseq_eval(outputs,labels):
    label_mask = (labels[:, :, 0] != -4).float()  # 使用第一个类别的填充值进行标签掩码
    masked_outputs = outputs * label_mask.unsqueeze(2)
    labels_outputs = labels * label_mask.unsqueeze(2)

    distance_total = 0
    for i in range(labels_outputs.shape[0]):
        predict = masked_outputs[i]
        label = labels_outputs[i]
        print(predict)
        predict_list = [torch.argmax(row).item() for row in predict if torch.max(row) != 0]
        label_list = [j for i, row in enumerate(label) for j, val in enumerate(row) if val == 1]
        distance = Levenshtein.distance(predict_list, label_list)
        distance_total = distance_total + distance
        print('---------------------------')
        print(predict_list)
        print(label_list)
    distance_aver = distance_total/labels_outputs.shape[0]
    print(f'平均编辑距离 {round(distance_aver,2)}')

def tensor_tolist(outputs,labels):
    label_mask = (labels[:, :, 0] != -4).float()  # 使用第一个类别的填充值进行标签掩码
    masked_outputs = outputs * label_mask.unsqueeze(2)
    labels_outputs = labels * label_mask.unsqueeze(2)
    label_lists = []
    predict_lists = []
    for i in range(labels_outputs.shape[0]):
        predict = masked_outputs[i]
        label = labels_outputs[i]
        predict_list = [torch.argmax(row).item() for row in predict if torch.max(row) != 0]
        label_list = [j for i, row in enumerate(label) for j, val in enumerate(row) if val == 1]
        label_lists.append(label_list)
        predict_lists.append(predict_list)

    return torch.tensor(predict_lists),torch.tensor(label_lists)

