# Revisiting Parameter-Efficient Tuning: Are We Really There Yet?

This is the code implementation of our paper accepted in EMNLP 2022:

> Guanzheng Chen, Fangyu Liu, Zaiqiao Meng, Shangsong Liang. [Revisiting Parameter-Efficient Tuning: Are We Really There Yet?](https://arxiv.org/abs/2202.07962).



We provide a comprehensive study for existing famous **P**arameter-**E**fficient **Tuning** (PETuning) methods, i.e., Adapter, Prompt, LoRA, and BitFit, focusing on their performance and stability.



The code structure is based in part on [P-tuning v2](https://github.com/THUDM/P-tuning-v2). (Thanks for their awesome work.)



## File Structure

- `model`: codes to implement PETuning methods.
- `tasks`: codes to preprocess datasets and choose model for each task.
- `training`: codes to define the trainer for training.
- `scripts`: scripts to run training, evaluation, and prediction for each task.
  - `search_scripts`: the scripts to perform grid search for each task.
  - `multiruns_scripts`: the scripts to conduct multi runs for each task, where each script contains the best hyper-parameters for corresponding task.
- `arguments.py & run.py`:  the arguments and running codes for training, evaluation, and prediction.



## Dependency

```
torch==1.8.1
transformers==4.5.0
adapter-transformers==2.2.0
```

Please view `requirements.txt` for more details.



## Data

All datasets in GLUE and SuperGLUE will be automatically downloaded (from Huggingface Datasets APIs) when running the scripts.



## PETuning for Each Task

To search the best hyper-parameters for each task,  you can run the scripts in the `scripts/search_scripts/` folder. For example,  you can run the CB tasks with adapter by the command:

```bash
bash scripts/search_scripts/superglue/search_adapter.sh cb
```



To conduct multiple runs for one task, you can run the scripts in the `scripts/multiruns_scripts/` folder.  For example,  you can run the CB tasks with adapter by the command:

```bash
bash scripts/multiruns_scripts/adapter/run_cb_roberta_both.sh
```

We provide the best hyper-parameters for each task in corresponding multi-runs scripts. If you cannot reproduce our reported results, please check the environment (package version) and conduct the grid search in your environment.



## Acknowledgments

[P-tuning v2](https://github.com/THUDM/P-tuning-v2)

[Hugging Face Transformers](https://github.com/huggingface/transformers)

[Adapter-Hub](https://github.com/Adapter-Hub/adapter-transformers)

[LoRA](https://github.com/microsoft/LoRA)

[BitFit](https://github.com/benzakenelad/BitFit)


## Citation

If you find our paper and resources useful, please kindly cite our paper:

```
@article{Chen2022RevisitingPT,
  title={Revisiting Parameter-Efficient Tuning: Are We Really There Yet?},
  author={Guanzheng Chen and Fangyu Liu and Zaiqiao Meng and Shangsong Liang},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.07962}
}
```
