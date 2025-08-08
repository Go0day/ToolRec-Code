import ast
import pickle
import json
import time
import gym
import requests
from bs4 import BeautifulSoup
from call_crs import retrieval_topk, stdout_retrived_items
from chat_api import llm_chat

from utils import *
import time


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class RecEnv(gym.Env):

    def __init__(self):
        """
        Initialize the environment.
        """
        super().__init__()
        self.obs = None  # current observation
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.observation_space = self.action_space = textSpace()
        self.num_retrieval = 0
        self.num_rerank = 0
        # self.send_chat = ssh_chat()
    
    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "recsys_steps":self.call_Recsys_cnt, "llm_steps":self.call_llm_cnt, "answer": self.answer, "rec_traj":[step for step in self.rec_traj]}

    def reset(self, seed=None, return_info=False, options=None, uid=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.obs = ("Interact with RecSys/LLM using retrieve[], rerank[], and finish[].\n")
        self.user_id = uid
        self.rec_traj = []
        self.page = None
        self.condition_keyword = None
        self.condition_list = None
        self.call_Recsys_cnt = 0
        self.call_llm_cnt = 0
        self.steps = 0
        self.answer = None
        self.final_length = 10
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def retrieval_step(self, attribute, topK):
        if attribute in DATASET_ATT:
            uid = [self.user_id]
            topk_score, external_item_list, external_item_list_name = retrieval_topk(dataset=dataset_name, condition=attribute, user_id=uid, topK=topK)
            retrived_items = stdout_retrived_items(topk_score, external_item_list, external_item_list_name)
            item_list = '[' + retrived_items[0] + ']'
            self.call_Recsys_cnt += 1
            self.rec_traj.append(['crs', topK, attribute, item_list])
            self.obs = item_list
        else:
            # retrieval from LLM, and or rerank....
            self.rerank_step(attribute, topK)

    
    def rerank_step(self, attribute, topK):
        instruction = prompt_pattern['knowledge_instruction_2']
        # examples = prompt_dict['ranking_sample']
        self.user_profile = user_profile[self.user_id]
        prompt_cur = ''
        for line in self.rec_traj:
            if line[0] == 'crs':
                prompt_cur = prompt_cur + prompt_pattern['crs_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
            elif line[0] == 'rerank':
                prompt_cur = prompt_cur + prompt_pattern['rerank_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
        last_mode = self.rec_traj[-1][0]
        if attribute == "None":
            previous_topK = sum([int(i[1]) for i in self.rec_traj])
            prompt_output = prompt_pattern['rerank_default_2'].format(before_topK=previous_topK, after_topK=topK)
        else:
            previous_topK = sum([int(i[1]) for i in self.rec_traj])
            prompt_output = prompt_pattern['rerank_output_2'].format(before_topK=previous_topK, rerank_type=attribute, after_topK=topK)
        
        question = user_profile[self.user_id] + prompt_cur + prompt_output
        attemps = 0
        while attemps < 10:
            attemps += 1
            try:
                # reranked_result = llm_chat(role=instruction, User_message=question)
                reranked_result = llm_chat(User_message=instruction+question)
                time.sleep(4)
                if not reranked_result.startswith("["):
                    reranked_result = '[' + reranked_result + ']'

                if extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                    break
            except Exception as e:
                time.sleep(5)
                print("An error occurred:", str(e))
        self.rec_traj.append(['rerank', topK, attribute, reranked_result])
        self.call_llm_cnt += 1
        self.obs = reranked_result

    def conclude_step(self, topK):
        instruction = prompt_pattern['knowledge_instruction_2']
        # examples = prompt_dict['ranking_sample']
        self.user_profile = user_profile[self.user_id]
        prompt_cur = ''
        for line in self.rec_traj:
            if line[0] == 'crs':
                prompt_cur = prompt_cur + prompt_pattern['crs_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
            elif line[0] == 'rerank':
                prompt_cur = prompt_cur + prompt_pattern['rerank_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
        previous_topK = sum([int(i[1]) for i in self.rec_traj])
        prompt_output = prompt_pattern['rerank_default_2'].format(before_topK=previous_topK, after_topK=topK)

        question = user_profile[self.user_id] + prompt_cur + prompt_output
        attemps = 0
        while attemps < 10:
            attemps += 1
            try:
                # reranked_result = llm_chat(role=instruction, User_message=question)
                reranked_result = llm_chat(User_message=instruction+question)
                time.sleep(4)
                if not reranked_result.startswith("["):
                    reranked_result = '[' + reranked_result + ']'
                if extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                    break
            except Exception as e:
                time.sleep(5)
                print("An error occurred:", str(e))
        
        self.rec_traj.append(['rerank', topK, ' ', reranked_result])
        self.call_llm_cnt += 1
        self.obs = reranked_result
        return reranked_result
            

    
    def step(self, action):
        '''retrieve[], rerank[], and finish[]'''
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()
    
        if action.startswith("retrieve[") and action.endswith("]"):
            rec_condition, topN = action[len("retrieve["):-1].split(',')[0].strip(), action[len("retrieve["):-1].split(',')[1].strip()
            topN = int(topN)
            self.retrieval_step(rec_condition, topN)
        elif action.startswith("rerank[") and action.endswith("]"):
            rec_condition, topN = action[len("rerank["):-1].split(',')[0].strip(), action[len("rerank["):-1].split(',')[1].strip()
            topN = int(topN)
            self.rerank_step(rec_condition, topN)
        elif action.startswith("finish"):
            answer = self.conclude_step(topK=self.final_length)
            self.answer = answer[1:-1]
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought.  But an Action cannot be a think."
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1

        return self.obs, reward, done, self._get_info()