import json
import pickle
import torch
from recbole.evaluator import Evaluator, Collector


dataset_name="ml-1m"
test_version="test/"

backbone_model="SASRec"
file_list = ['XXX']  # saved text file name

model_file_dict = {
    'SASRec': {
        'ml-1m': {
            'None': 'SASRec-XXXX.pth',
            'genre': 'SASRec_AddInfo2-XXXX.pth',
            'release_year': 'SASRec_AddInfo2-XXXX.pth',
        },
        'amazon_book': {
            'None': 'SASRec-XXXX.pth',
            'price': 'SASRec_AddInfo2-XXXX.pth',
            'sales_rank': 'SASRec_AddInfo2-XXXX.pth',
        },
        'yelp': {
            'None': 'SASRec-XXXX.pth',
            'city': 'SASRec_AddInfo2-XXXX.pth',
            'stars': 'SASRec_AddInfo2-XXXX.pth',
            'categories': 'SASRec_AddInfo2-XXXX.pth',
        }},
    'BERT4Rec': {
        'ml-1m': {
            'None': 'BERT4Rec-XXXX.pth',
            'genre': 'BERT4Rec_AddInfo-XXXX.pth',
            'release_year': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'None': 'BERT4Rec-XXXX.pth',
            'price': 'BERT4Rec_AddInfo-XXXX.pth',
            'sales_rank': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'None': 'BERT4Rec-XXXX.pth',
            'city': 'BERT4Rec_AddInfo-XXXX.pth',
            'stars': 'BERT4Rec_AddInfo-XXXX.pth',
            'categories': 'BERT4Rec_AddInfo-XXXX.pth',
        }
    }
}

model_BERT = {
    'SASRec': {
        'ml-1m': {
            'genre': 'SASRec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'sales_rank': 'SASRec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'categories': 'SASRec_AddInfo-XXXX.pth',
        }},
    'BERT4Rec': {
        'ml-1m': {
            'None': 'BERT4Rec-XXXX.pth',
            'genre': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'None': 'BERT4Rec-XXXX.pth',
            'sales_rank': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'None': 'BERT4Rec-XXXX.pth',
            'categories': 'BERT4Rec_AddInfo-XXXX.pth',
        }
    }
}


DATASET_ATT = model_file_dict[backbone_model][dataset_name].keys()

prompts_path = './dataset/prompts/' + test_version
# prompts_path5 = './dataset/prompts/length5/'
checkpoint_path = './dataset/saved_file/'

prompt_file = dataset_name + '_ICL.json'
profile = dataset_name + '_chat.pkl'
prompt_pattern = dataset_name + '_pattern.json'

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

# with open(prompts_path5 + profile, 'rb') as f:
#     uid_iid, user_profile, item_profile, itemID_name = pickle.load(f)

with open(prompts_path + profile, 'rb') as f:
    uid_iid, user_profile, item_profile, itemID_name = pickle.load(f)

with open(prompts_path + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

with open(prompts_path + prompt_pattern, 'r') as f:
    prompt_pattern = json.load(f)


token_path = prompts_path + dataset_name + '_ui_token.pkl'
with open(token_path, 'rb') as f:
    user_token_id, user_id_token, item_token_id, item_id_token = pickle.load(f)

# knowledge_prompt = prompt_dict['Reranking']


def extract_user_reclist(ranked_str):
    uid_topK = {}
    for uid, item_str in ranked_str.items():
        uid_topK[uid] = [item.split(', ')[0] for item in item_str[1:-1].strip().split('\n')]
    return uid_topK

def cleaning_user_itemList(ui_dict, topk=10):
    larger_than = 0
    smaller_than = 0
    uid_list = [u for u in ui_dict.keys()]
    for uid in uid_list:
        if len(ui_dict[uid]) == 10:
            continue
        elif len(ui_dict[uid]) > 10:
            larger_than += 1
            ui_dict[uid] = ui_dict[uid][:10]
        elif len(ui_dict[uid]) < 10:
            smaller_than += 1
            del ui_dict[uid]
    return ui_dict, larger_than, smaller_than

def check_itemList_length(ui_dict, topk=10):
    res_right = 1
    uid_list = [u for u in ui_dict.keys()]
    for uid in uid_list:
        if len(ui_dict[uid]) == 10:
            continue
        else:
            res_right = 0
    return res_right


def extract_and_check_cur_user_reclist(ranked_str, topk=10):
    ranked_str = ranked_str[1:-1]
    res_right = 1
    cur_user_reclist = [item.split(', ')[0] for item in ranked_str.strip().split('\n')]
    if len(cur_user_reclist) == topk:
        # check length fits top K
        res_right = 0
    for iid in cur_user_reclist:
        if not item_token_id.get(iid, 0):
        # check item range is suitable.
            res_right = 0
    return res_right

def evaluate_user(user_id, pos_item, user_topK, user_num, item_num):
    topk_idx = torch.tensor(user_topK)
    positive_u = torch.tensor(user_id)     # minus 1 to ensure the matrix follows 0 -> user_num
    positive_i = torch.tensor(pos_item)

    # user_num = 943
    # item_num = 1683
    pos_matrix = torch.zeros((user_num, item_num), dtype=torch.int)
    pos_matrix[positive_u, positive_i] = 1
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
    result_matrix = torch.cat((pos_idx, pos_len_list), dim=1)
    data_struct = {}
    data_struct["rec.topk"] = result_matrix

    # Evaluate
    config = {}
    config["metric_decimal_place"] = 4
    config['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
    config['topk'] = [10]
    evaluator = Evaluator(config)

    result = evaluator.evaluate(data_struct)
    return result

def statstics_recedItems_LLM(uid_topK, top=10):
    item_freq = {}
    for u in uid_topK:
        for iid in uid_topK[u]:
            if item_freq.get(iid, 0):
                item_freq[iid] += 1
            else:
                item_freq[iid] = 1
    item_name_list = [ii for ii in item_freq.keys()]
    item_freq_list = [item_freq[ii] for ii in item_name_list]
    zipped_lists = zip(item_name_list, item_freq_list)
    # Sort the zipped lists based on the values in list2 in descending order
    sorted_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    # Unzip the sorted lists
    sorted_list1, sorted_list2 = zip(*sorted_lists)
    # Retrieve the top 2 elements
    top_item_name = sorted_list1[:top]
    top_item_freq = sorted_list2[:top]
    return top_item_name, top_item_freq

def statstics_recedItems_CRS(uid_topK, top=10):
    item_freq = {}
    for i_list in uid_topK:
        for iid in i_list:
            if item_freq.get(iid, 0):
                item_freq[iid] += 1
            else:
                item_freq[iid] = 1
    item_name_list = [ii for ii in item_freq.keys()]
    item_freq_list = [item_freq[ii] for ii in item_name_list]
    zipped_lists = zip(item_name_list, item_freq_list)
    # Sort the zipped lists based on the values in list2 in descending order
    sorted_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    # Unzip the sorted lists
    sorted_list1, sorted_list2 = zip(*sorted_lists)
    # Retrieve the top 2 elements
    top_item_name = sorted_list1[:top]
    top_item_freq = sorted_list2[:top]
    return top_item_name, top_item_freq