# Code for paper "P$^3$SUM: Preserving Author’s Perspective in News Summarizationwith Diffusion Language Models" accepted at NAACL@2024!

## Acknowledgement
The diffusion part heavily relies on [TESS](https://github.com/allenai/tess-diffusion) and [SSD-LM](https://github.com/xhan77/ssd-lm)  We extend our sincere gratitude to the authors for generously sharing the code in advance and for their exceptional work. Please check their remarkable papers as well:
1. [TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379)
2. [SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control](https://arxiv.org/abs/2210.17432)

## Installation

```sh
conda create -n preserve python=3.8.5
conda activate preserve
conda env update --file env.yml
```
## Run
First fill the path and other hyperparameters in ```simplex-diffusion-main/commands_for_sum/run_sum.sh```:
```sh
cd "path/to/simplex-diffusion-main"
model_name="roberta-base"
model_path="model/" #path to the finetuned diffusion model for summarization task
learning_rate=3e-5
max_steps=200000 #120000
CUDA_VISIBLE_DEVICES=1
datasetname="Sampled_Datasets/cnn_dm_500_raw" #path to the input datasets
dataset_config_name="3.0.0"
cache_dir="./"
preprocessing_num_workers=16
overwrite_cache=false
per_device_train_batch_size=8
per_device_eval_batch_size=16
do_train=false #whether train the model
do_eval=false
do_predict=true #wether test the model(generate summaries)
evaluation_strategy="no"
eval_steps=1000
report_to="tensorboard"
overwrite_output_dir=false
max_seq_length=512
max_target_length=100 # recommend to be close to the avg. length of gold summary 
max_source_length=412 #num of tokens in news context
val_max_target_length=100
skip_special_tokens=true
max_eval_samples=100
max_predict_samples=500
simplex_value=5
num_diffusion_steps=1000
lr_scheduler_type="linear"
pad_to_max_length=true
beta_schedule="squaredcos_improved_ddpm"
weight_decay=0.0
warmup_steps=2000
max_steps=200000
gradient_accumulation_steps=1
logging_steps=50
save_steps=20000
conditional_generation="ul2"
save_total_limit=1
tokenized_data_path="raw/xsum/roberta"
metric_for_best_model="rouge1"
if_control=true
ctr_model_name='POLITICS_model' #path to the off-the-shelf classifier
#ctr_model_name=None
ctr_opt_label_idx=0 #political leaning of the input news context
output_dir="out/" #path to save generated summaries
decode_ctr_lr=1000
```
## Resources
We have made the [Sampled_Datasets](https://drive.google.com/file/d/1qIYjVl9wI-BYYO9C1XRgOsgEyFm-Xd7o/view?usp=sharing) available online for your convenience. For the off-the-shelf classifier, we recommend reaching out to the authors of the [POLITICS paper](https://arxiv.org/abs/2205.00619). If you need the weights of the diffusion model for summarization task, please feel free to email Yuhan Liu at lyh6560@stu.xjtu.edu.cn.

## Cite
```
@inproceedings{liu-etal-2024-p3sum,
    title = "{P}$^3${S}um: Preserving Author{'}s Perspective in News Summarization with Diffusion Language Models",
    author = "Liu, Yuhan  and
      Feng, Shangbin  and
      Han, Xiaochuang  and
      Balachandran, Vidhisha  and
      Park, Chan Young  and
      Kumar, Sachin  and
      Tsvetkov, Yulia",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.119",
    pages = "2154--2173",
    abstract = "In this work, we take a first step towards designing summarization systems that are faithful to the author{'}s intent, not only the semantic content of the article. Focusing on a case study of preserving political perspectives in news summarization, we find that existing approaches alter the political opinions and stances of news articles in more than 50{\%} of summaries, misrepresenting the intent and perspectives of the news authors. We thus propose P$^3$Sum, a diffusion model-based summarization approach controlled by political perspective classifiers. In P$^3$Sum, the political leaning of a generated summary is iteratively evaluated at each decoding step, and any drift from the article{'}s original stance incurs a loss back-propagated to the embedding layers, steering the political stance of the summary at inference time. Extensive experiments on three news summarization datasets demonstrate that P$^3$Sum outperforms state-of-the-art summarization systems and large language models by up to 13.7{\%} in terms of the success rate of stance preservation, with competitive performance on standard metrics of summarization quality. Our findings present a first analysis of preservation of pragmatic features in summarization, highlight the lacunae in existing summarization models{---}that even state-of-the-art models often struggle to preserve author{'}s intents{---}and develop new summarization systems that are more faithful to author{'}s perspectives.",
}

```