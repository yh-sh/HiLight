# HiLight
Learning Evidence Highlighting for Frozen LLMs

# Running the `schedule` Slurm job (with Conda + requirements.txt)

This guide shows how to:
1) create the `llm_emph` conda environment
2) install Python packages from `requirements.txt`  
3) authenticate to Hugging Face safely (no hard-coded tokens)  
4) submit the Slurm job

---

## 0) Prerequisites

- You have Miniconda/Anaconda installed (or the cluster provides it).
- You have the datasets downloaded.
- You can download models from Huggingface.


> **Important security note:** do **NOT** commit or hard-code your Hugging Face token in a job script. Use an environment variable instead.

```bash
# required for gated models (e.g. Llama 3, Gemma 3); sbatch passes it on to the job
export HF_TOKEN=<your_hf_token>
```

---

## 1) Create and activate the Conda environment

### Create a fresh env
```bash
# load conda into your shell (adjust path if needed)
source ~/miniconda3/etc/profile.d/conda.sh

# create the env (pick the python version your project expects)
conda create -n llm_emph python=3.12 -y
conda activate llm_emph

python -m pip install --upgrade pip

pip3 install torch torchvision

# install your dependencies
pip install -r requirements.txt
```
### Use existing env (Only if you cannot create your own env)
```bash
# load module you have, which should be a conda env and a slurm module
module load *your_env_name*

python -m pip install --upgrade pip

pip3 install torch torchvision

# install your dependencies
pip install -r requirements.txt

# change the lines in .job file
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_emph
# to
module load *your_env_name*
```
## 2) Download data

Download the dataset files from the following Google Drive links:

- https://drive.google.com/file/d/12nolJpnDnicrEntiNFch74YGxoVbYmNJ/view?usp=sharing  
- https://drive.google.com/file/d/14Ya2m9i3nVGv3TMHYasn3weiGTMhHNH5/view?usp=sharing  
- https://drive.google.com/file/d/1BZLX5FVWlBzqrbY8wz9Kx_EZ3CeFdp8P/view?usp=sharing  
- https://drive.google.com/file/d/1FUuhPOaDbBbkZGvmxLz-lerTi3ZfMUpP/view?usp=sharing  
- https://drive.google.com/file/d/1hK7tbqRN8JAyhxCVD0ePM9e9u9bRRAL4/view?usp=sharing  
- https://drive.google.com/file/d/1koqkQwnSUYJUzvA2mDX-2olOKT4EAmko/view?usp=sharing  

After downloading, place **all** files into the `data/` folder.

## 3a) Run the bash jobs (Bash only)

```bash
nohup bash submit_grid.sh > logs/rec/grid.nohup.out 2>&1 &
disown
```

## 3b) Run the slurm jobs
```bash
chmod +x submit_grid_slurm.sh scripts/run_one.sbatch
./submit_grid_slurm.sh
```

## 4 a) Test the checkpoint

Change the model-id (Main_LLM) and actor_param (the checkpoint of actor) in **schedule.job**.
Then,
```bash
sbatch schedule.job
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026learning,
  title     = {Learning Evidence Highlighting for Frozen LLMs},
  author    = {Li, Shaoang and Shi, Yanhang and Li, Yufei and Liang, Mingfu and Wei, Xiaohan and Pu, Yunchen and Tian, Fei and Sun, Chonglin and Shyu, Frank and Pandey, Sandeep and Simon, Luke and Liu, Xi and Li, Jian},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
