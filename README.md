# Let Me Do It For You: Towards LLM-Empowered Recommendation via Tool Learning

This project builds upon the powerful attribute-based retrieval capabilities of [RecBole](https://github.com/RUCAIBox/RecBole), a comprehensive, flexible, and easy-to-use recommendation algorithm library.  
We further integrate a **User Decision Simulation** module with the **ReAct pipeline**, enhancing the system's decision-making process for recommendation tasks.

---

## Features
- **Sequential Recommendation Pretraining** using ConditionSASRec.
- **Attribute-Oriented Retrieval Tool** construction with partial finetuning.
- **ToolRec** module for LLM-empowered recommendation.
- **Evaluation Pipeline** for recommendation performance analysis.

---

## 1. Pretrain Sequential Recommendation
Prepare the base sequential recommender for attribute retrieval tools.

```bash
# Prepare dataset info
python run_recbole.py --model=SASRec --dataset=ml-1m --config_files=SAS_ml1m.yaml --dump_to_chat=True --chat_hislen=10 --seed=2023 --test_v=test

python run_recbole.py --model=SASRec --dataset=ml-1m --config_files=SAS_ml1m.yaml --dump_profile=True --test_v=test

# Sequential recommendation example
python run_recbole.py --model=SASRec --dataset=ml-1m --config_files=SAS_ml1m.yaml --dump_profile=False --gpu_id=0 --test_v=test
```

> **Note:** Update the saved model name in the YAML file:  
>
> ```yaml
> pretrained_name: /SASRec-XXXX.pth
> ```

---

## 2. Prepare Attribute-Oriented Retrieval Tool
Partly train on the base RecTool and save model checkpoints.

```bash
python run_recbole.py --dataset=ml-1m --model=SASRec_AddInfo --config_files=SAS_ml1m.yaml --gpu_id=0 --show_progress=False --load_pretrain=True --freeze_Rec_Params=True --item_additional_usage=True --item_additional_feature=genre --side_feature_size=100 --bert_usage=True --test_v=test --pretrained_name=/SASRec-XXXX.pth
```

> **Note:** Save all fine-tuned file names in `utils.py` under `class dataset_sideinfo`.  
> For example, for **ml-1m**, include `None`, `genre`, and `release_year` as mentioned in the paper.

---

## 3. Start ToolRec
Set the configuration in `utils.py`:

```python
dataset_name = "ml-1m"
test_version = "test"
backbone_model = "SASRec"
```

Run ToolRec:

```bash
nohup python chat_RecAct.py > SASRec_ml-1m_toolrec.txt 2>&1 &
```

---

## 4. Evaluate Recommendation Performance
Edit `chat_analysis.py` to specify the file list:

```python
file_list = ['SASRec_ml-1m_toolrec']  # saved nohup text file name
```

Run the evaluation:

```bash
python chat_analysis.py
```

---

## File Setup
**YAML configs:**
- `amazon-book.yaml`
- `SAS_ml1m.yaml`
- `yelp_rec.yaml`

**Python files:**
- `utils.py`
- `chat_api.py` (requires OpenAI API key)

**Downloads:**
- `dataset/glove/glove.6B.100d.txt`

---

## Dataset
- This repo contains **ml-1m** in `./datasets/`.
- To use the original dataset, replace the item IDs using:
  ```bash
  python replace_item_ID.py
  ```
- Alternatively, download the processed dataset from our Google Cloud link (TBA).

---

## Acknowledgments
This project was made possible thanks to the contributions and inspiration from the following works:

- **[RecBole](https://github.com/RUCAIBox/RecBole)** — A comprehensive, flexible, and easy-to-use recommendation algorithm library, which provides the foundation for our attribute-based retrieval capabilities.
- **[ReAct](https://github.com/reactjs)** — A framework whose pipeline inspired the integration of our User Decision Simulation module, enhancing decision-making processes in recommendation tasks.


---

## Contact
This code was written quite some time ago and is preserved in its original state for reference only. It has not been updated or maintained recently.  
If you have any questions, feel free to contact via email:

📧 **yuyuezha00@gmail.com**
