# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
from logging import getLogger

import pandas as pd


import pickle
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import random


def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if config['dump_to_chat']:
        dump_userInfo_chat(config['test_v'], config['dataset'], test_data, his_len=config['chat_hislen'])
        sys.exit()
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

def dump_userInfo_chat(test_v, dataset, test_data, his_len=10):
    uid_iid = {}
    uid_iid_his = {}
    uid_iid_hisScore = {}
    data = test_data.dataset.inter_feat.interaction
    for (uid, iid, iid_his, i_len, iid_hisScore) in zip(data['user_id'], data['item_id'], data['item_id_list'], data['item_length'], data['rating_list']):
        u_token = test_data.dataset.id2token('user_id', [uid])[0]
        i_token = test_data.dataset.id2token('item_id', [iid])[0]
        iid_his_token = test_data.dataset.id2token('item_id', iid_his)

        uid_iid[u_token] = i_token
        if i_len >= his_len:
            uid_iid_his[u_token] = iid_his_token[i_len - his_len:i_len]
            uid_iid_hisScore[u_token] = iid_hisScore[i_len - his_len:i_len]
        else:
            uid_iid_his[u_token] = iid_his_token[:i_len]
            uid_iid_hisScore[u_token] = iid_hisScore[:i_len]
    
    users = list(uid_iid.keys())
    sampled_users = random.sample(users, 200)
    uid_iid_small = {u: uid_iid[u] for u in sampled_users}
    uid_iid_his_small = {u: uid_iid_his[u] for u in sampled_users}
    uid_iid_hisScore_small = {u: uid_iid_hisScore[u] for u in sampled_users}

    # file_path = './dataset/prompts/' + dataset + '_uid_dict.pkl'
    file_path = './dataset/prompts/' + test_v + '/' + dataset + '_uid_dict.pkl'
    # with open(file_path, 'wb') as f:
    #     pickle.dump((uid_iid, uid_iid_his, uid_iid_hisScore), f)
    with open(file_path, 'wb') as f:
        pickle.dump((uid_iid_small, uid_iid_his_small, uid_iid_hisScore_small), f)
    user_token_id = test_data.dataset.field2token_id['user_id']
    item_token_id = test_data.dataset.field2token_id['item_id']
    user_id_token = test_data.dataset.field2id_token['user_id']
    item_id_token = test_data.dataset.field2id_token['item_id']

    token_path = './dataset/prompts/' + test_v + '/' + dataset + '_ui_token.pkl'
    with open(token_path, 'wb') as f:
        pickle.dump((user_token_id, user_id_token, item_token_id, item_id_token), f)
    raise KeyboardInterrupt
    




def run_recboles(rank, *args):
    ip, port, world_size, nproc, offset = args[3:]
    args = args[:3]
    run_recbole(
        *args,
        config_dict={
            "local_rank": rank,
            "world_size": world_size,
            "ip": ip,
            "port": port,
            "nproc": nproc,
            "offset": offset,
        },
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def save_split_dataset(dataset, split=None, topK=1):
    '''
    Top K means remain the most or second most attribute for each user.
    ...... to construct the 
    '''
    if split == 'class':
        item_df = dataset.item_feat
        item_df['class'] = item_df['class'].apply(lambda x: ','.join(str(x)))
    else:
        item_df = dataset.item_feat
    group_df = dataset.inter_feat.groupby(dataset.uid_field)
    

    group_list = list(group_df)
    for uid, udf in group_list:
        udf = pd.merge(udf, item_df, on=dataset.iid_field)
        udf_split_topK = udf.groupby(split)[split].count().sort_values().index[-topK:].values
        # write dataframe rows into files.  which in udf_split_topK.
        for index, row in udf.iterrows():
            if row[split] in udf_split_topK:
                pass


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
