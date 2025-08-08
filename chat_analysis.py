import json
import ast
import pickle
import torch
from recbole.evaluator import Evaluator, Collector
import random
from call_crs import retrieval_topk

import logging
from utils import *

if dataset_name == 'yelp':
    item_num = 45582
    # user_num = 1003
elif dataset_name == 'ml-1m':
    item_num = 3884  
elif dataset_name == 'amazon_book':
    item_num = 97566



step_all = 0
rec_call = 0
llm_call = 0

user_list = [uid for uid in uid_iid.keys()]





def assign_random_topK(item_range, exclude=None, topk=10):
    sampled_numbers = []
    while len(sampled_numbers) < topk:
        sample_tmp = random.randint(1, item_range)
        sample_tmp = str(sample_tmp)
        if sample_tmp not in exclude:
            sampled_numbers.append(sample_tmp)
    return sampled_numbers

def assign_random_iid(item_range, exclude=None):
    while True:
        sample_tmp = random.randint(1, item_range)
        sample_tmp = str(sample_tmp)
        if sample_tmp not in exclude:
            return sample_tmp

def not_all_elements_are_digits(lst):
    for element in lst:
        if not element.isdigit():
            return True
    return False

def get_user_rec_list(infos_list, topk=10):
    '''
    chat Recommendation list.
    '''
    uid_topK = {}
    all_user_num = 0
    valid_num = 0
    for info in infos_list:
        for user_info in info:
            # if user_info["user_idx"] == '741':
            #     print(123)
            uid_topK[user_info["user_idx"]] = []
            res_tmp = [item.split(', ')[0] for item in user_info['answer'].strip().split('\n')]
            if len(res_tmp) == topk:
                valid_num += 1
                uid_topK[user_info["user_idx"]] = res_tmp
            else:
                # select the topK item result from recommendation trajectory
                for tj in user_info['rec_traj'][::-1]:
                    if str(tj[1]).isnumeric() and int(tj[1]) >= topk:
                        uid_topK[user_info["user_idx"]] = [item.split(",")[0] for item in tj[3].strip()[1:-1].strip().split("\n")][:topk]
                        if len(uid_topK[user_info["user_idx"]]) == topk:
                            break
            if len(uid_topK[user_info["user_idx"]]) < topk or not_all_elements_are_digits(uid_topK[user_info["user_idx"]]):
                del uid_topK[user_info["user_idx"]]
                # uid_topK[user_info["user_idx"]] = assign_random_topK(item_range=item_num-1, exclude=uid_iid_his[user_info["user_idx"]], topk=10)
            all_user_num += 1
    print("Chat: All User num is: {all}, Valid User num is {valid}, Invalid Num is {invalid}.".format(all=all_user_num, valid=valid_num, invalid=all_user_num-valid_num ))

    return uid_topK

def get_chat_rec_list(infos_list, topk=10):
    '''
    chat Recommendation list.
    '''
    uid_topK = {}
    for info in infos_list:
        for user_info in info:
            res_tmp = [item.split(', ')[0] for item in user_info['answer'].strip().split('\n')]
            if len(res_tmp) >= topk:
                uid_topK[user_info["user_idx"]] = res_tmp[:topk]
                if not_all_elements_are_digits(uid_topK[user_info["user_idx"]]):
                    del uid_topK[user_info["user_idx"]]
    return uid_topK


def get_user_crs_list(infos_list, topk=10):
    uid_topK = {}
    all_num = 0
    invalid_num = 0
    for info in infos_list:
        for user_info in info:
            uid_topK[user_info["user_idx"]] = []
            for tj in user_info['rec_traj']:
                if tj[0] == 'crs' and int(tj[1]) >= topk:
                    uid_topK[user_info["user_idx"]] = [item.split(",")[0] for item in tj[3].strip()[1:-1].strip().split("\n")][:topk]
            if len(uid_topK[user_info["user_idx"]]) < topk:
                invalid_num += 1
                # uid_topK[user_info["user_idx"]] = assign_random_topK(item_range=item_num-1, exclude=uid_iid_his[user_info["user_idx"]], topk=10)
            all_num += 1
    print("CRS: All User num is: {all}, Valid User num is {valid}, Invalid Num is {invalid}.".format(all=all_num, valid=all_num-invalid_num, invalid=invalid_num ))
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



uid_topK = {}
for idx in file_list:
    with open('chat_his/{dataset}/{index}.txt'.format(dataset=dataset_name, index=idx), 'r') as file:
        for line in file:
            if line[:8] == "{'steps'":
                line_new = line.strip()
                line_dict = ast.literal_eval(line_new)
                uid_topK[line_dict["user_idx"]] = [item.split(', ')[0] for item in line_dict['answer'].strip().split('\n')]
                step_all += line_dict["steps"] - 1
                rec_call += line_dict["recsys_steps"]
                llm_call += line_dict["llm_steps"]
ui_dict, larger_than, smaller_than = cleaning_user_itemList(uid_topK)
# Analysis with Text







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

# ================================================================================
# ================================================================================
count_out_index = 0

uid_topK_tmp = uid_topK.copy()
for uid in uid_topK_tmp:
    for iidx, iid in enumerate(uid_topK_tmp[uid]):
        if not item_token_id.get(iid, 0):
            count_out_index += 1
            del uid_topK[uid]
            break
            # uid_topK[uid][iidx] = assign_random_iid(item_range=item_num-1, exclude=uid_iid_his[uid])

# ============================
print("Out index Num is: {out_idx}".format(out_idx=count_out_index))

print("step: {step},  rec_call: {rec},  llm_call: {llm}".format(step=step_all, rec=rec_call, llm=llm_call))



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

top_item_name, top_item_freq = statstics_recedItems_LLM(uid_topK)

user_num = len(uid_topK)
pos_user_before_map = [user_token_id[uid] for uid in uid_topK.keys()]
pos_user_before_map.sort()
pos_user_list_str = [user_id_token[uid] for uid in pos_user_before_map]

# pos_user_list = pos_user_before_map
pos_user_list = [i for i in range(user_num)]
pos_item_list = [item_token_id[uid_iid[uid]] for uid in pos_user_list_str]
topk_idx_list = [[item_token_id[iid] for iid in uid_topK[uid]] for uid in pos_user_list_str]
chat_eval_result = evaluate_user(pos_user_list, pos_item_list, topk_idx_list, user_num, item_num)
print('ToolRec: ')
print(chat_eval_result)

