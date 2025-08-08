
import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model
import pickle
import numpy as np
from recbole.utils import get_trainer
from utils import *



def retrieval_topk(dataset, condition='None', user_id=None, topK=10, mode='freeze'):
    model_name = model_file_dict[backbone_model][dataset][condition]
    if mode != 'freeze':
        model_name = model_BERT[backbone_model][dataset][condition]
    model_file = checkpoint_path + model_name
    
    # load trained model
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file,
    )
    
    # retrieval top K items, and the corresponding score.
    uid_series = dataset.token2id(dataset.uid_field, user_id)

    topk_score, topk_iid_list = full_sort_topk(
        uid_series, model, test_data, k=topK, device=config["device"]
    )
    # print(topk_score)  # scores of top 10 items
    # print(topk_iid_list)  # internal id of top 10 items
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    # print(external_item_list)
    external_item_list_name = []
    for u_list in external_item_list:
        external_item_list_name.append([itemID_name[iid] for iid in u_list])
    external_item_list_name = np.array(external_item_list_name)


    return topk_score, external_item_list, external_item_list_name

def stdout_retrived_items(score, item_id, item_name):
    retrived_items = []
    for n in range(item_id.shape[0]):
        item_strings = ""
        for s, iid, ina in zip(score[n], item_id[n], item_name[n]):
            item_strings = item_strings + str(iid) + ', ' + str(ina) + ", " + str(round(s.item(), 4)) + "\n"
        retrived_items.append(item_strings)
    return retrived_items

    
# if __name__ == "__main__":
    
#     # test

#     # score = full_sort_scores(uid_series, model, test_data, device=config["device"])
#     # print(score)  # score of all items
#     # print(
#     #     score[0, dataset.token2id(dataset.iid_field, ["242", "302"])]
#     # )  # score of item ['242', '302'] for user '196'.
#     users = ["8", "88", "588", "688", "888"]
#     topK = 6
#     topk_score, external_item_list, external_item_list_name = retrieval_topk(condition='ne', user_id=users, topK=topK)
#     retrived_items = stdout_retrived_items(topk_score, external_item_list, external_item_list_name)

#     topk_score1, external_item_list1, external_item_list_name1 = retrieval_topk(condition='None', user_id=users, topK=topK)
#     retrived_items1 = stdout_retrived_items(topk_score1, external_item_list1, external_item_list_name1)