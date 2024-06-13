from src.utils.utils_io_model import *
from src.hunman_doc.constant import *
from src.utils.utils_dataloader import *
from src.utils.utils_dataloader import *
import time


def get_virdiag_model(fn_pkl,flag,model_sort,diag_modelname):
    data_copy = Data(fn=fn_pkl, flag=flag)
    data_copy = feature_scaling(data_copy)
    data_copy = randomdoc_to_xgb(data_copy)

    _, best_auc1, model = best_auc_model(model_path, flag, diag_modelname)
    randomdoc_diag = model.predict(data_copy.embedding)
    sum_diag = sum(randomdoc_diag)
    print(f'random doctor diag num is :{len(randomdoc_diag)}')



def virdoc_first_diag():
    fn_pkl = randomdoc_model_input_path + f'first_data_dim_{feature_dimnum_first}_randomdoc.pkl'
    get_virdiag_model(fn_pkl,'first','diag',first_diag_modelname)




def virdoc_final_diag():
    fn_pkl = randomdoc_model_input_path + f'/final_data_dim_{feature_dimnum_final}_randomdoc.pkl'
    get_virdiag_model(fn_pkl,'final','diag',final_diag_modelname)



if __name__ == '__main__':
    # start_time = time.time()
    # virdoc_first_diag()
    # end_time = time.time()
    # elapsed_minutes = (end_time - start_time) / 60
    # print(f"初步诊断 程序运行时间: {elapsed_minutes:.2f} 分钟，也就是{(end_time - start_time)}秒")


    start_time = time.time()
    virdoc_final_diag()
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"最终诊断 程序运行时间: {elapsed_minutes:.2f} 分钟，也就是{(end_time - start_time)}秒")