# Evaluating domain-specific LLMs and prompt formats

## What's in the box?
- demo_utils: utility code for running the evaluation notebook
- prompt_evaluation_demo.ipnb: notebook demonstrating evaluation concepts and generates datasets with model responses.
- blog_toy_dataset.csv: Hand-curated and labeled dataset for demonstrating the concepts covered in the notebook above.


## Setup
```
conda create --name llm_evaluation_demo python=3.11
conda activate llm_evaluation_demo
pip install poetry==1.8.3
conda install -c "nvidia/label/cuda-12" cuda-toolkit
# pass the `--no-root` flag because this project doesn't have a root directory. This suppresses the associated warning. 
poetry install --no-root
```

Set your huggingface API token in the environment.
```
export HF_TOKEN=<your token goes here.>
```

Ensure that you've got access to the model we're using for this demonstration.

Go to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 and request access. 

If you skip this step, you won't be able to download and load the base model into memory. 

## Environment
This demo was developed using the huggingface framework for working Transformer models, with CUDA 12 tooling, and 
on [NVIDIA v100 hardware](https://www.nvidia.com/en-gb/data-center/tesla-v100/).

You'll need at least 16 gigs of GRAM to run this particular model in inference mode.
