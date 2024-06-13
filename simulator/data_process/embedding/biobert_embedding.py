from transformers import AutoTokenizer, AutoModel
import numpy as np
from project_path import *

#在haggingface上下的预训练模型
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
biobert_model = AutoModel.from_pretrained(biobert_path)


def get_biobert_embeddings(text):
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()
    return embeddings, hidden_embeddings


def split_note_document(text, min_length=15):

    tokens_list_0 = biobert_tokenizer.tokenize(text)
    if len(tokens_list_0) <= 510:
        return [text], [1]

    chunk_parse = []
    chunk_length = []
    chunk = text
    ## Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = biobert_tokenizer.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = biobert_tokenizer.tokenize(temp[i])
            if len_sub + len(temp_tk) > 510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else:
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))
    # print("Parsed lengths: ", chunk_length)

    return chunk_parse, chunk_length

#event_weights 按照 检查时间距离 入院时间的 间隔（小时为单位）
def biobert_embedding(full_events_list, event_weights):
    event_weights_exp = []
    for idx, event_string in enumerate(full_events_list):
        weight = event_weights[idx]
        string_list, lengths = split_note_document(event_string)
        for idx_sub, event_string_sub in enumerate(string_list):
            # Extract biobert embedding
            embedding, hidden_embedding = get_biobert_embeddings(event_string_sub)
            # Concatenate
            if (idx == 0) & (idx_sub == 0):
                full_embedding = embedding
            else:
                full_embedding = np.concatenate((full_embedding, embedding), axis=0)
            event_weights_exp.append(weight)

    try:
        aggregated_embedding = np.average(full_embedding, axis=0, weights=np.array(event_weights_exp))
    except:
        aggregated_embedding = np.zeros(768)

    return aggregated_embedding


if __name__ == '__main__':
    text1 = '胸部前后位摄影 (AP CHEST FILM), 于凌晨2点12分进行。临床指征: 新安置的鼻胃管 (nasogastric tube) ，评估其位置。与患者前一次于17:05进行的影像对照。凌晨2点12分进行的便携式直立位胸部前后位摄影。'
    text2 = '一周前心肌梗死，慢性阻塞性肺疾病。有中心静脉插管和气管插管。印象:与12:41 a.m.比较，AP胸片:新术后气管插管管尖端位于锁骨上缘，与气管分叉至少7厘米，并应进一步推进3厘米以确保更加稳定。'

    text = [text1,text2]

    #检查时间距离入院时间的时间间隔列表，按小时计算
    detaladmittime = [2,3]

    aggregated_embedding = biobert_embedding(text,detaladmittime)
    aggregated_embedding = aggregated_embedding.reshape(1,-1)
    print(aggregated_embedding.shape)
