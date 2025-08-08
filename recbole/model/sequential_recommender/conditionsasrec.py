# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""
import copy
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerLayer_Prefix_Adapter
from recbole.model.loss import BPRLoss



class ConditionSASRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ConditionSASRec, self).__init__(config, dataset)
         # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.bert_usage = config["bert_usage"]
        self.item_additional_feature = config["item_additional_feature"]
        self.item_additional_usage = config["item_additional_usage"]
        self.item_transformer_mode = config["item_transformer_mode"]
        # define layers and loss

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Condition Parameters
        self.adapter_prefix_usage = config["adapter_prefix_usage"]
        self.prefix_hidden_size = config["prefix_hidden_size"]
        self.prefix_len = config["prefix_len"]
        self.dim_additional_feature = config['dim_additional_feature']
        self.prefix_embedding = nn.Embedding(self.prefix_hidden_size, self.hidden_size, padding_idx=0)
        self.prefix_q_embedding = nn.Embedding(self.prefix_hidden_size, self.hidden_size, padding_idx=0)

        self.adapter1=torch.nn.Sequential( nn.LayerNorm(self.hidden_size),
                                nn.Linear(self.hidden_size,self.hidden_size//4),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size//4,self.hidden_size),
                               
                                # nn.LayerNorm(self.hidden_size)
                )
        self.adapter2=torch.nn.Sequential(nn.LayerNorm(self.hidden_size),
                                nn.Linear(self.hidden_size,self.hidden_size//4),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size//4,self.hidden_size),
                                # nn.LayerNorm(self.hidden_size)
                )
        if config['item_transformer_layer'] == 1:
            self.item_feature_transform=torch.nn.Sequential(nn.LayerNorm(self.hidden_size + self.dim_additional_feature),
                                    nn.Linear(self.hidden_size + self.dim_additional_feature, self.hidden_size),
                                    nn.ReLU(),
                    )
        elif config['item_transformer_layer'] == 2:
            self.item_feature_transform=torch.nn.Sequential(nn.LayerNorm(self.hidden_size + self.dim_additional_feature),
                                    nn.Linear(self.hidden_size + self.dim_additional_feature, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.ReLU(),
                    )
        elif config['item_transformer_layer'] == 3:
            self.item_feature_transform=torch.nn.Sequential(nn.LayerNorm(self.hidden_size + self.dim_additional_feature),
                                    nn.Linear(self.hidden_size + self.dim_additional_feature, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.ReLU(),
                    )
        self.item_interaction_feature = dataset.item_feat[config['item_additional_feature']].to(self.device)
        if len(self.item_interaction_feature.shape) < 2:
            self.item_interaction_feature = self.item_interaction_feature.unsqueeze(1)
        # if self.dim_additional_feature == 1:
        #     self.item_interaction_feature = self.item_interaction_feature.unsqueeze(1)
        
        self.SASRecEncoder = SASRecEncoder(config, dataset)
        
        self.apply(self._init_weights)

        if config["load_pretrain"]:
            pretrained_path = config["checkpoint_dir"] + config["pretrained_name"]
            pretrained_parameter = torch.load(pretrained_path)["state_dict"]
            current_state_dict = self.state_dict()
            for key in list(pretrained_parameter.keys()):
                if key not in current_state_dict:
                    del pretrained_parameter[key]
            current_state_dict.update(pretrained_parameter)
            self.load_state_dict(current_state_dict)

        
        if config["freeze_Rec_Params"]:
            for p_name, param in self.named_parameters():
                param.requires_grad = False
                if 'adapter' in p_name or 'prefix' in p_name or 'item_feature_transform' in p_name:
                    param.requires_grad = True


    def forward(self, item_seq, item_seq_len, item_additional_embedding=None):
        prefix=None
        adapter=None
        item_feature_transform = None
        batch_size, seq_len = item_seq.shape

        if self.adapter_prefix_usage:
            prefix=[list(range(1+i*self.prefix_len,1+(i+1)*self.prefix_len)) for i in range(self.n_layers)]
            prefix=[prefix]*batch_size
            prefix=torch.tensor(prefix).transpose(1,2)
            prefix=prefix.to(item_seq.device)
            prefix_k=self.prefix_embedding(prefix)
            prefix_q=self.prefix_q_embedding(prefix)
            prefix=prefix_q, prefix_k
            adapter=[self.adapter1,self.adapter2]
        
        if self.item_additional_usage:
            item_feature_transform = self.item_feature_transform

        # SASRec Encoder
        output = self.SASRecEncoder(item_seq, item_seq_len, item_additional_embedding, adapter, prefix, item_feature_transform)

        # Seq Embedding mixed with feature embedding.
        return output 
    

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_additional_embedding = None
        if self.item_transformer_mode == 1:
            item_additional_embedding = interaction[self.item_additional_feature]
        elif self.item_transformer_mode == 3:
            item_additional_embedding = self.item_interaction_feature
        seq_output = self.forward(item_seq, item_seq_len, item_additional_embedding=item_additional_embedding)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.SASRecEncoder.item_embedding(pos_items)
            neg_items_emb = self.SASRecEncoder.item_embedding(neg_items)
            if self.item_feature_transform and self.item_transformer_mode == 2:
                pos_items_additional_feature = self.item_interaction_feature[pos_items]
                neg_items_additional_feature = self.item_interaction_feature[neg_items]
                pos_items_emb = self.item_feature_transform(torch.cat((pos_items_emb, pos_items_additional_feature), dim=1))
                neg_items_emb = self.item_feature_transform(torch.cat((neg_items_emb, neg_items_additional_feature), dim=1))
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.SASRecEncoder.item_embedding.weight
            if self.item_feature_transform and self.item_transformer_mode == 2:
                test_item_emb = self.item_feature_transform(torch.cat((test_item_emb, self.item_interaction_feature), dim=1))
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_additional_embedding = None
        if self.item_transformer_mode == 1:
            item_additional_embedding = interaction[self.item_additional_feature]
        elif self.item_transformer_mode == 3:
            item_additional_embedding = self.item_interaction_feature
        seq_output = self.forward(item_seq, item_seq_len, item_additional_embedding=item_additional_embedding)
        test_item_emb = self.SASRecEncoder.item_embedding(test_item)
        if self.item_feature_transform and self.item_transformer_mode == 2:
            test_items_additional_feature = self.item_interaction_feature[test_item]
            test_item_emb = self.item_feature_transform(torch.cat((test_item_emb, test_items_additional_feature), dim=1))
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_additional_embedding = None
        if self.item_transformer_mode == 1:
            item_additional_embedding = interaction[self.item_additional_feature]
        elif self.item_transformer_mode == 3:
            item_additional_embedding = self.item_interaction_feature
        seq_output = self.forward(item_seq, item_seq_len, item_additional_embedding=item_additional_embedding)
        test_item_emb = self.SASRecEncoder.item_embedding.weight
        if self.item_feature_transform and self.item_transformer_mode == 2:
            test_item_emb = self.item_feature_transform(torch.cat((test_item_emb, self.item_interaction_feature), dim=1))
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B n_items]
        return scores



class SASRecEncoder(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecEncoder, self).__init__(config, dataset)
         # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        # Layers and 
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)


        layer = TransformerLayer_Prefix_Adapter(
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.transformer_block = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.n_layers)])

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.item_transformer_mode = config["item_transformer_mode"]

    def forward(self, item_seq, item_seq_len, item_additional_embedding=None, adapter=None, prefix=None, item_feature_transform=None):
        batch_size, seq_len = item_seq.size(0), item_seq.size(1)
        position_ids_range = torch.arange(
            seq_len, dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids_range.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        if item_feature_transform and self.item_transformer_mode == 3:
            input_seq_additional_feature_emb = item_additional_embedding[item_seq]
            input_emb = torch.cat((input_emb, input_seq_additional_feature_emb), dim=-1)
            input_emb = item_feature_transform(input_emb)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        all_encoder_layers = []
        
        for layer, block in enumerate(self.transformer_block):

            if adapter is not None and prefix is None:
                input_emb = block.forward_adapter(input_emb, extended_attention_mask, adapter[0],adapter[1])
            elif adapter is not None and prefix is not None:
                input_emb = block.forward_adapter_prefix(input_emb, extended_attention_mask, adapter[0],adapter[1], prefix)
            elif adapter is None and prefix is not None:
                input_emb = block.forward_prefix(input_emb, extended_attention_mask, adapter[0],adapter[1], prefix)
            else:
                input_emb = block(input_emb, extended_attention_mask)
            all_encoder_layers.append(input_emb)

        output = all_encoder_layers[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        if item_feature_transform and self.item_transformer_mode == 1:
            output = torch.cat((output, item_additional_embedding), dim=1)
            output = item_feature_transform(output)

        return output  # [B H]


