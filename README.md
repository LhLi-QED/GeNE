# Context Length Extension via Generalized Extrapolation Scale

**Ge**neralized extrapolatio**N** scal**E** (GeNE), a straightforward and effective method applied to the interpolate function of positional embeddings to achieve training short, test long. Experimental results show
that GeNE notably improves long context language modeling. By randomly scaling the extrapolation ratio during the finetuning, GeNE achieves stable extrapolation on 64k contexts by training on 16k length.

## Data
We uploaded our training and test data to the Google cloud, and these files can be downloaded from [this link](https://drive.google.com/drive/folders/1WS-xo9XAAry7_MA4y0aClz6TGnbaA1Bp?usp=sharing) and need to be manually placed in the `data` directory

## Model

We've released our models on huggingface that are finetuned on $\leq$ 16k length of text.

| Model | Link  |
|-------|-------|
|llama2-gene-64k-base| [Huggingface 🤗](https://huggingface.co/voolxanQED/llama2-gene-64k-base)|
|llama2-gene-64k-chat| [Huggingface 🤗](https://huggingface.co/voolxanQED/llama2-gene-64k-chat)|






## Usage

Take the example of finetuning Llama2 checkpoint, use the following command:

```shell
torchrun --nproc_per_node 8 finetune.py --training_config_path ./configs/ptrain_16k_64k.yaml
```
Custom training can be achieved by modifying the config files in `configs`.

Run the following command to evaluate the PPL of the model:
```shell
cd ./evaluation
deepspeed --num_gpus 8 ppl.py --model_path path-to-your-checkpoint --tokenizer_path path-to-tokenizer
```

To evaluate the accuracy of passkey retrieval, use the following command:
```shell
deepspeed --num_gpus 4 passkey.py --model_path path-to-your-checkpoint --tokenizer_path path-to-tokenizer
```