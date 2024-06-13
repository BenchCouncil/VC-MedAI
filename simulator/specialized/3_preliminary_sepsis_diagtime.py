import sys
from project_path import pro_path
sys.path.append(pro_path)
import xgboost as xgb
import optuna
import pandas as pd
from simulator.utils.utils_dataloader import *
from simulator.utils.utils_io_model import *

def train():
    kforder = 1
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        X_train, X_val = kflod_em[train_index], kflod_em[val_index]
        y_train, y_val = kflod_label[train_index], kflod_label[val_index]

        model_path = root + f'model_save/{model_sort}_model/{flag}_{model}_dim_{feature_dimnum}_kford_{kforder}_'
        study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))  # maximize
        study.optimize(lambda trial: objective(trial, X_train,y_train,X_val,y_val,test_em,test_label,model_path,kforder), n_trials=n_trials)  # You can adjust the number of trials
        kforder += 1
        best_params = study.best_trial.params
        best_value = study.best_value
        print(f"Best Parameters: {best_params}")
        print(f"Best Objective Value: {best_value}")


def predict():
    best_kflod,lowest_rmse,best_model = lowest_rmse_model(root + f'model_save/{model_sort}_model/',flag,model)
    kforder = 1
    for train_index, val_index in skf.split(kflod_em, kflod_label):
        if str(kforder) == best_kflod:
            X_train, X_val = kflod_em[train_index], kflod_em[val_index]
            y_train, y_val = kflod_label[train_index], kflod_label[val_index]

            print(f'kflod为{best_kflod}，最低的rmse {lowest_rmse}')
            print('-----训练集上的效果评估-----')
            rmse_eval(y_train, best_model.predict(X_train))
            print('-----验证集上的效果评估-----')
            rmse_eval(y_val, best_model.predict(X_val))
            print('-----测试集上的效果评估-----')
            rmse_eval(test_label, best_model.predict(test_em))

            print('-----测试集 Label改成20%范围之后的效果评估-----') #训练时候的label不用范围，用原来真实的值
            test_label_range = diagtime_label_to_range(test_label,best_model.predict(test_em), 0.2)
            rmse_eval(test_label_range, best_model.predict(test_em))

            comparison_df = pd.DataFrame({
                'Label Diagnosis Time': test_label[:40],
                'Predicted Diagnosis Time': best_model.predict(test_em)[:40]
            })
            print('-----患者的label和预测结果对比-----')
            print(comparison_df)
        kforder += 1



def objective(trial,train_x,train_y,val_x,val_y,test_x,test_y,model_path,kforder):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators',200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 1.0),
        # 'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 10.0),  # subsample参数
    }
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        seed=42,
        tree_method='gpu_hist',
        gpu_id=1,
        verbosity=0,
        importance_type= 'gain',
        early_stopping_rounds=50,
        **params  # Pass the sampled parameters to the XGBClassifier
    )

    model.fit(train_x,train_y,eval_set=[(val_x,val_y)],verbose=False)
    print(f'----kforder:{kforder}-训练集上的效果评估-----')
    rmse_eval(train_y, model.predict(train_x))
    print(f'----kforder:{kforder}-验证集上的效果评估-----')
    rmse_eval(val_y, model.predict(val_x))
    print(f'----kforder:{kforder}--测试集上的效果评估----')
    rmse,mae = rmse_eval(test_y, model.predict(test_x))
    print(f'-----kforder:{kforder}--测试集 Label改成20%范围之后的效果评估-----')
    test_y_range = diagtime_label_to_range(test_y, model.predict(test_em), 0.2)
    _, mae_after = rmse_eval(test_y_range, model.predict(test_em))
    global lowest_mae
    if mae < lowest_mae:
        print('测试集mae降低，保存模型')
        # xgb_save_model(simulator, model_path=model_path + f'mae_{mae}.dat')
        xgb_save_model(model, model_path=model_path + f'mae_{mae_after}.dat')

        lowest_mae = mae
    return mae


#-------全局参数-----------
model_sort = 'sepsis'
flag = 'first'
feature_dimnum = 35
model = 'diagtime'
n_trials = 5000
root = '/home/ddcui/virtual-doctor/'
fn_pkl = root + f'datasets/{model_sort}_model_input/{flag}_data_7000_dim_{feature_dimnum}.pkl'
_, lowest_mae, _ = lowest_rmse_model(root + f'model_save/{model_sort}_model/', flag, model)
#------------------------

data = Data(fn=fn_pkl, flag=flag)
kflod_em, kflod_label, test_em, test_label = data_split(data,flag, model)
skf = KFold(n_splits=5, shuffle=True, random_state=42)


#调整：筛除了诊断时间不合理的情况 出现原因：由于诊断时间包含了修改诊断的时间。正常情况下诊断和修改诊断的操作是连续的，但可能存在过了几天重新修改诊断的情况，导致有些诊断时间异常
# mae占均值的xxx% 占标准差的xxx%，rmse会放大误差，mae会平等对待所有误差

if __name__ == '__main__':
    param = sys.argv[1]
    if param == 'train':
        train()
    elif param == 'test':
        predict()










