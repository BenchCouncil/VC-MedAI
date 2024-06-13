import sys
from project_path import pro_path
sys.path.append(pro_path)
from simulator.specialized.click_sequence.datasets import *
from simulator.specialized.click_sequence.trs_constant import *
from simulator.specialized.click_sequence.data_load_test import *
from simulator.specialized.click_sequence.transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import warnings
# 忽略特定类型的警告
warnings.filterwarnings("ignore")
from generator.random_virdoc import uuid_randomdoc_dict


def get_dec_input(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input




def preprocess_emb_randomdoc():
    fn_pkl =  f'{pro_path}datasets/sepsis_model_input/first_data_7000_dim_35_clickseq.pkl'
    data_true = Data(fn=fn_pkl, flag='first')
    data = ramdom_doc(data_true) #替换为随机医生的信息

    data.embedding = data.embedding.astype(int)
    arr = data.embedding
    arr_integer = np.floor(np.abs(arr) * 10)
    value_to_index = {}
    index = 0
    for i, row in enumerate(arr_integer):
        for j, element in enumerate(row):
            if element not in list(value_to_index.keys()):
                value_to_index[element] = index
                index += 1
    arr_integer_index = np.zeros_like(arr_integer)
    for i in range(arr_integer.shape[0]):
        for j in range(arr_integer.shape[1]):
            arr_integer_index[i, j] = value_to_index[arr_integer[i, j]]
    data.embedding = arr_integer_index
    return data


def add_nextact(data_new):
    i = 1
    for uuid,emb in zip(data_new.uuid,data_new.embedding):
        i+=1
        enc_input = torch.LongTensor(emb)
        enc_input = enc_input.to(device)
        predict_dec_input = get_dec_input(model, enc_input.view(1, -1), start_symbol=tgt_vocab["S"])
        predict, _, _, _ = model(enc_input.view(1, -1), predict_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]

        non_zero_predict = predict[(predict != 0)]
        if 2 in non_zero_predict:
            index_2 = (non_zero_predict == 2).nonzero(as_tuple=False)[0]
            non_zero_predict = non_zero_predict[:index_2]
        predict_str_list = [idx2word[n.item()] for n in non_zero_predict]
        count = sum(1 for elem in predict_str_list if '下一步' in elem)
        nextact = count/8
        print(f'{i},{uuid}:{nextact}')
        data_to_save = (uuid, nextact)
        with open(topath, 'ab') as file:
            pickle.dump(data_to_save, file)


def read_nextact_pkl(topath):
    global uuid_list,nextact_list
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


def ramdom_doc(data1):
    data_copy = copy.deepcopy(data1)
    # 替换医生信息为随机医生信息
    for uuid, embedding in zip(data_copy.uuid, data_copy.embedding):
        if uuid in uuid_randomdoc_dict.keys():
            randomdoc_emb = uuid_randomdoc_dict.get(uuid)
            embedding[1:8] = randomdoc_emb  # 只修改医生的信息
            data_copy.embedding[data1.uuid.index(uuid)] = embedding  # 不直接使用data是因为，data被修改了 缩放到了0-1
    return data_copy

topath = f'{pro_path}model_save/randomdoc_nextact/sepsis_model_ramdom_doctor_nextact_predict.pkl'


if __name__ == '__main__':
    model = Transformer().to(device)
    model.load_state_dict(torch.load(model_path+test_model_name))
    model.eval()

    #预测下一步检查百分比
    data_new = preprocess_emb_randomdoc()
    add_nextact(data_new)

