import os
import sys


sys.path.insert(0,os.path.join('/project/venv/lib/python3.8/site-packages/'))
sys.path.insert(0,os.path.join('/venv/lib/python3.8/site-packages'))

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
import numpy as np
import pandas as pd
import dill
from ipynb.fs.full.uaspeech_dataset import *

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")


def create_split(ds):
    severities = {
        "train": {
            "m": [],
            "h": [],
            "l": [],
            "v_l": [],
            "all": [],
            "c": [],
            "speakers_v_l": [],
            "transcripts_v_l": [],
            "speakers_l": [],
            "transcripts_l": [],
            "speakers_m": [],
            "transcripts_m": [],
            "speakers_h": [],
            "transcripts_h": [],
            "speakers_all": [],
            "transcripts_all": [],
            "severities_all": [],
            "transcripts_c": [],
            "speakers_c": [],
            "block_id_all": [],
            "block_id_c": [],
            "block_id_v_l": [],
            "block_id_l": [],
            "block_id_m": [],
            "block_id_h": [],
        },
        "test": {
            "m": [],
            "h": [],
            "l": [],
            "v_l": [],
            "all": [],
            "c": [],
            "speakers_v_l": [],
            "transcripts_v_l": [],
            "speakers_l": [],
            "transcripts_l": [],
            "speakers_m": [],
            "transcripts_m": [],
            "speakers_h": [],
            "transcripts_h": [],
            "speakers_all": [],
            "transcripts_all": [],
            "severities_all": [],
            "transcripts_c": [],
            "speakers_c": [],
            "block_id_all": [],
            "block_id_v_l": [],
            "block_id_l": [],
            "block_id_m": [],
            "block_id_h": [],
        },
    }

    for i, item in enumerate(ds):
        if item["block_id"] == "B2" and item["severity"]!= "c":
            split = "test"
        else:
            split = "train"
        severities[split][item["severity"]].append(item["audio"])
        severities[split][f"speakers_{item['severity']}"].append(item["speaker_id"])
        severities[split][f"transcripts_{item['severity']}"].append(item["transcription"])
        severities[split][f"block_id_{item['severity']}"].append(item["block_id"])

    sevs = ["v_l", "l", "m", "h"]
    for split in ["test", "train"]:
        for sev in sevs:
            severities[split]["all"].extend(severities[split][sev])
            severities[split]["speakers_all"].extend(severities[split][f"speakers_{sev}"])
            severities[split]["transcripts_all"].extend(severities[split][f"transcripts_{sev}"])
            severities[split]["block_id_all"].extend(severities[split][f"block_id_{sev}"])
            severities[split]["severities_all"].extend(np.full(len(severities[split][f"transcripts_{sev}"]), sev))

    print("save splits")
    with open('/datasets/split.pkl', 'wb') as outp:
        dill.dump(severities, outp)


def find_paths(ds, inds, split):
    attributes = ["speakers", "block_id", "transcripts"]
    infos = {
        "speakers": [],
        "block_id": [],
        "transcripts": []
    }

    for attribute in attributes:
        if split == "train":
            info = ds[f"{attribute}_all"] + ds[f"{attribute}_c"]
        elif split == "test":
            info = ds[f"{attribute}_all"]
        info = list(map(lambda i: info[i], inds))
        infos[attribute] = info

    word_filename = pd.read_excel("/data/project/uafiles/doc/speaker_wordlist.xls", sheet_name="Word_filename")

    # find paths
    paths = []
    for i in range(len(inds)):
        filename = word_filename[word_filename["WORD"] == infos["transcripts"][i]]["FILE NAME"].values
        if len(filename) > 1:
            # some transcripts are used multiple times -> find correct one
            fn = [s for s in filename if infos["block_id"][i] in s]
            filename = fn[0]
        elif len(filename) == 1:
            filename = filename[0]
        else:
            print("error while searching for filename")

        # combine parts of path
        if filename.startswith("B"):  # uncommon word
            path = infos["speakers"][i] + "_" + filename
        else:
            path = infos["speakers"][i] + "_" + infos["block_id"][i] + "_" + filename

        paths.append(path)

    return paths

def create_csv(task):
    with open(f"datasets/split.pkl", 'rb') as inp:
        ds = dill.load(inp)

    splits = ["train", "test"]
    for split in splits:
        if split == "train":
            id = range(len(ds[split]["all"])+len(ds[split]["c"]))
        elif split == "test":
            id = range(len(ds[split]["all"]))
        wav_path = find_paths(ds[split], id, split)
        wav_path = ["/data/project/uaspeech/audiofinal/"+path+".wav" for path in wav_path]

        if task == "asr":
            if split == "train":
                transcription = ds[split]["transcripts_all"] + ds[split]["transcripts_c"]
            elif split == "test":
                transcription = ds[split]["transcripts_all"]
            df = pd.DataFrame(list(zip(id,wav_path, transcription)), columns=["id", "wav_path", "transcription"])
        elif task == "sid":
            if split == "train":
                label = ds[split]["speakers_all"] + ds[split]["speakers_c"]
            elif split == "test":
                label = ds[split]["speakers_all"]
            df = pd.DataFrame(list(zip(id, wav_path, label)), columns=["id", "wav_path", "label"])
        elif task == "sev":
            if split == "train":
                label = ds[split]["severities_all"] + list(np.repeat("c", len(ds[split]["c"])))
            elif split == "test":
                label = ds[split]["severities_all"]
            df = pd.DataFrame(list(zip(id, wav_path, label)), columns=["id", "wav_path", "label"])


        df.to_csv(f"/datasets/{split}_{task}.csv", index=False)



def create_ds(ds, name, remove):
    severities = {
        "m" : [],
        "h" : [],
        "l" : [],
        "v_l" : [],
        "all" : [],
        "c": [],
        "speakers_v_l" : [],
        "transcripts_v_l" : [],
        "speakers_l": [],
        "transcripts_l": [],
        "speakers_m": [],
        "transcripts_m": [],
        "speakers_h": [],
        "transcripts_h": [],
        "speakers_all": [],
        "transcripts_all": [],
        "severities_all": [],
        "transcripts_c": [],
        "speakers_c": [],
        "block_id_all": [],
        "block_id_v_l": [],
        "block_id_l": [],
        "block_id_m": [],
        "block_id_h": [],
        "block_id_c": [],
    }

    crayon = {
        "v_l": [],
        "m": [],
        "h": [],
        "l": [],
        "all": [],
        "severities": [],
        "speakers_v_l": [],
        "speakers_l": [],
        "speakers_m": [],
        "speakers_h": [],
        "speakers_all": [],
        "speakers_c": [],
        "c": [],
    }

    for i, item in enumerate(ds):
        #print("i",i)
        severities[item["severity"]].append(item["audio"])
        severities[f"speakers_{item['severity']}"].append(item["speaker_id"])
        severities[f"transcripts_{item['severity']}"].append(item["transcription"])
        severities[f"block_id_{item['severity']}"].append(item["block_id"])

        if item["transcription"] == "crayon":
            crayon[item['severity']].append(item["audio"])
            crayon[f"speakers_{item['severity']}"].append(item["speaker_id"])


    sevs = ["v_l", "l", "m", "h"]
    for sev in sevs:
        severities["all"].extend(severities[sev])
        severities["speakers_all"].extend(severities[f"speakers_{sev}"])
        severities["transcripts_all"].extend(severities[f"transcripts_{sev}"])
        severities["block_id_all"].extend(severities[f"block_id_{sev}"])
        severities["severities_all"].extend(np.full(len(severities[f"transcripts_{sev}"]), sev))

        crayon["all"].extend(crayon[sev])
        crayon[f"speakers_{sev}"].extend(crayon[f"speakers_{sev}"])


    print("save datasets")
    with open(name, 'wb') as outp:
        dill.dump(severities, outp)


if __name__ == "__main__":

    # correct data here
    with open("/data/project/uaspeech/ua_ds_merged.pkl", 'rb') as inp: # use merged data
        ds_all = dill.load(inp)

    # correct name
    name = "/datasets/severities_merged.pkl"

    #with open(f"/data/project/uaspeech/ua_ds_{argv[1]}.pkl", 'rb') as inp: # use M2
    #   ds_all = dill.load(inp)

    #name=f"/project/thesis/datasets/{argv[1]}.pkl"

    print("create datasets")
    create_ds(ds_all, name, False)

    print("create split")
    create_split(ds_all)

    #create_csv("asr")
    #create_csv("sid")
    #create_csv("sev")




