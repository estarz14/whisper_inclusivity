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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor
import dill
from pandas.core.common import flatten
from sys import argv

def load_pkl(file_name):
    with open(file_name, 'rb') as inp:
        data = dill.load(inp)

    return data

def transcribe(inputs, model, processor):
    with torch.no_grad():
        print("predict")
        n_samples = len(inputs)
        transcripts = []

        for i in range(1, 11):  # predict 10 subsets of samples
            print(i)
            ind_before = int((i - 1) * (n_samples / 10))
            ind = int(i * (n_samples / 10))

            # predict ids
            pred = model.generate(inputs[ind_before:ind])
            # decode ids to transcripts
            transcripts.append(processor.batch_decode(pred, skip_special_tokens=True))

            del pred
    transcripts = list(flatten(transcripts))
    print("it worked")
    return transcripts

def count_transcripts(transcripts):
    char_count = [len(t) for t in transcriptions]
    word_count = [len(t.split()) for t in transcriptions]

    counts = {
        "char_count": char_count,
        "word_count": word_count,
    }

    return counts


if __name__ == '__main__':
    sr = 16000

    print("load model")
    ## Load model & Feature Extractor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", attn_implementation="eager")
    model_ft = WhisperForConditionalGeneration.from_pretrained("/model/whisper-base-ft/checkpoint-21", attn_implementation="eager")

    for m in [model, model_ft]:
        m.generation_config.language = "english"
        m.generation_config.task = "transcribe"
        m.generation_config.is_multilingual = False
        m.generation_config.forced_decoder_ids = None

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    # load dataset
    print("load dataset")
    if argv[1] == "merged":
        ds = load_pkl(f"/datasets/severities_merged.pkl")
    elif argv[1] == "test":
        ds = load_pkl(f"/datasets/severities_merged_test.pkl")
    elif argv[1] == "unmerged":
        ds = load_pkl(f"/datasets/severities.pkl")
    elif argv[1] == "M2":
        ds = load_pkl(f"/datasets/severities_M2.pkl")
    elif argv[1] == "M3":
        ds = load_pkl(f"/datasets/severities_M3.pkl")
    elif argv[1] == "M4":
        ds = load_pkl(f"/datasets/severities_M4.pkl")
    elif argv[1] == "M5":
        ds = load_pkl(f"/datasets/severities_M5.pkl")
    elif argv[1] == "M6":
        ds = load_pkl(f"/datasets/severities_M6.pkl")
    elif argv[1] == "M7":
        ds = load_pkl(f"/datasets/severities_M7.pkl")
    elif argv[1] == "M8":
        ds = load_pkl(f"/datasets/severities_M8.pkl")
    elif argv[1] == "final":
        ds = load_pkl(f"/datasets/final.pkl")
    else:
        print("wrong modus")
        print(argv[1])
    ds_subsets = ds["all"]+ds["c"]


    # extract features
    print("extract features")
    inputs = feature_extractor(ds_subsets, sampling_rate=sr, do_normalize=True,
                                            return_tensors="pt").input_features

    # transcribe all samples
    print("transcribe inputs")
    transcriptions = transcribe(inputs, model, processor)

    # count samples
    print("count samples")
    counts = count_transcripts(transcriptions)

    with open(f'/datasets/counts_{argv[1]}_pt.pkl', 'wb') as outp:
        dill.dump(counts, outp)

    with open(f'/datasets/transcripts_{argv[1]}_pt.pkl', 'wb') as outp:
        dill.dump(transcriptions, outp)





