import os
import sys

sys.path.insert(0, os.path.join('/project/venv/lib/python3.8/site-packages/'))
sys.path.insert(0, os.path.join('/venv/lib/python3.8/site-packages'))


def ignore_user_installs(username):
    ## avoid using user installs
    user_install_path = '/scratch/' + username + '/python/lib/python3.8/site-packages'
    if user_install_path in sys.path:
        sys.path.remove(user_install_path)


ignore_user_installs("starzew")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import Dataset
import dill
import evaluate
from transformers import TrainerCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer, EvalPrediction
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from dataclasses import dataclass
from typing import Any, Dict, List, Union


class UADataset(Dataset):
    def __init__(self, audios, labels):
        self.audios = audios
        self.labels = labels

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        audio = processor(self.audios[index], sampling_rate=16000).input_features[0]
        label = self.labels[index]

        return audio, label


def load_pkl(file_name):
    with open(file_name, 'rb') as inp:
        data = dill.load(inp)

    return data


def prepare_data(audio, processor):
    input_features = processor(audio, sampling_rate=16000).input_features[0]

    return input_features


def prepare_labels(label, processor):
    target = processor(text=label).input_ids

    return target

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        #print(f"features: {features}")
        input_features = [{"input_features": feature[0]} for feature in features]
        #input_features = features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature[1]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # print("labels: ", label_str)

    # normalize preds & labels
    normalizer = BasicTextNormalizer()
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return wer
    # return {"wer": wer}


# code taken from https://stackoverflow.com/questions/68617212/passing-two-evaluation-datasets-to-huggingface-trainer-objects
class MetricCollection:
    def __init__(self, dataset_1_size):
        self.dataset_1_size = dataset_1_size

    def mymetric(self, pred):
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # print("labels: ", label_str)

        # normalize preds & labels
        pred_str = [processor.tokenizer._normalize(pred) for pred in pred_str]
        label_str = [processor.tokenizer._normalize(label) for label in label_str]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return wer

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        metrics = {}
        #print("labelids", p.label_ids.__len__())

        labels_1 = p.label_ids[:self.dataset_1_size]
        labels_2 = p.label_ids[self.dataset_1_size:]

        predictions_1 = p.predictions[:self.dataset_1_size, :]
        predictions_2 = p.predictions[self.dataset_1_size:, :]

        p.label_ids = labels_1
        p.predictions = predictions_1

        metrics['wer_uaspeech'] = self.mymetric(p)

        p.label_ids = labels_2
        p.predictions = predictions_2

        metrics['wer_librispeech'] = self.mymetric(p)

        return metrics

# https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/7
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


if __name__ == "__main__":
    # Load model & Feature Extractor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    # Load data
    ds_ua = load_pkl('/datasets/split.pkl')
    ds_ua["train"]["transcripts_all"] = list(
        map(lambda x: prepare_labels(x, processor), ds_ua["train"]["transcripts_all"]))
    ds_ua["test"]["transcripts_all"] = list(
        map(lambda x: prepare_labels(x, processor), ds_ua["test"]["transcripts_all"]))

    # Libri speach test clean for eval in finetuning
    ls_test = load_pkl("/data/project/librispeech/test-clean.pkl")
    ls_test["transcripts"] = list(map(lambda x: prepare_labels(x, processor), ls_test["transcripts"]))

    train_data = UADataset(ds_ua["train"]["all"], ds_ua["train"]["transcripts_all"])
    # append libri speach test clean to ua speech test set
    test_data = UADataset(ds_ua["test"]["all"] + ls_test["audio"],
                          ds_ua["test"]["transcripts_all"] + ls_test["transcripts"])


    # Generation Config
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.is_multilingual = False

    model.generation_config.forced_decoder_ids = None

    # Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metric
    metric = evaluate.load("wer")
    metric_calculator = MetricCollection(len(ds_ua["test"]["transcripts_all"]))

    # Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir="/model/whisper-base-ft",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        max_steps=30,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1,
        eval_steps=1,
        logging_steps=1,
        # report_to=["tensorboard"],
        load_best_model_at_end=False,
        #metric_for_best_model="wer_uaspeech",
        greater_is_better=False,
        push_to_hub=False,
    )

    # trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        compute_metrics=metric_calculator.compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.add_callback(EvaluateFirstStepCallback())

    # training
    print("start training")
    history = trainer.train()

    with open(f'/model/log_hist_30.pkl', 'wb') as outp:
        dill.dump(trainer.state.log_history, outp)

    # evaluation
    #evaluation = trainer.evaluate()

    print("save model")
    # save model
    model.save_pretrained("/model/base-ft")
