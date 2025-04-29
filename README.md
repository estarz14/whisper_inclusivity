# whisper_inclusivity
Implementation of the experiments in my master thesis "Investigating Inclusivity of Pre-Trained Speech Recognition Model Whisper for Pathological Speech"


## Data Preprocessing & Data Selection

- notebooks/uaspeech_dataset.ipynb: initial steps of processing, microphone and gender statistics, preparation of librispeech test-clean for fine-tuning

- pipelines/preprocess_datasets.py: create final dataset, create training split, create csv files for modified superb tasks
    - python3 preprocess_datasets.py [mic] -> which dataset to choose


- pipelines/hist_pipeline.py
    - Transcribes selected dataset & counts number of transcribed words
    - python3 hist_pipeline.py [mic] -> which dataset to choose

- notebooks/data_preprocessing.ipynb: dataset statistics, data selection, effects of normalization and trimming


## Fine-Tuning

- pipelines/ft_pipeline.py: fine-tune Whisper following the HuggingFace tutorial https://huggingface.co/blog/fine-tune-whisper

- notebooks/finetune_whisper.ipynb: visualization of fine-tuning performance

## Experiments

### Representation Comparison

- pipelines/precompute_predictions.py: calculate embedding-based representations (both variants tavg/favg), attention-based representations, downsampled attentions for cumatt 

- notebooks/att_representation.ipynb: experiment on how to calculate attention-based representations

- notebooks/emb_representation.ipynb: experiment on how to calculate attention-based representations, datasplit experiment, t-sne visualizations

- pipelines/compare_representations.py
    - compare representations between healthy and dysarthric samples, pre-trained and fine-tuned representations with the four selected metrics
    - comparison for the block selection experiment

- notebooks/Results_Comparison.ipynb: visualization of results

### Representational Capacity

- experiments are condcuted with the help of s3prl (SUPERB Benchmark) -> needs to be installed first

- the folder whisper has to be added to the cloned s3prl/upstream

- in s3prl/upstream/interfaces.py, lines 257-259 have to be commented out

- for details on how to run the experiments, have a look at the official documentation https://github.com/s3prl/s3prl/tree/main

- modified superb tasks: asr_ua.py, pr_ua.py, sev_ua.py, sid_ua.py
- python3 [asr_ua.py] --target_dir [result/exp1] --prepare_data.dataset_root datasets --build_upstream.name [whisper_pt]

- notebooks/Results_Probing.ipynb: visualization of results

- the computed files by superb tasks are not uploaded because of memory constraints

### Head Categorization

- pipelines/head_pipeline.py
    - calculate head scores
    - rank & categorize heads
    - create head masks using the categorizations (layer-wise and global)
    - head ablation experiment (layer-wise and global)

- notebooks/Results_Head_Cat.ipynb: visualization of results

### Temporal Attention Behavior

- attention values are saved by pipelines/precompute_predictions.py (Section Representation Comparison)

- notebooks/Results_Temp_Attention.ipynb: visualization of temporal attention
