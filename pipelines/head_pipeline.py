import os
import sys
import torch
import numpy as np
from scipy.stats import entropy
from transformers import WhisperForConditionalGeneration, AutoFeatureExtractor
#from preprocess_datasets import load_dataset
import pandas as pd
import dill
import yaml
from itertools import combinations

import subprocess

sys.path.insert(0,os.path.join('/project/venv/lib/python3.8/site-packages/'))
sys.path.insert(0,os.path.join('/venv/lib/python3.8/site-packages'))

def ignore_user_installs(username):
    ## avoid using user installs
    user_install_path = '/scratch/' + username + '/python/lib/python3.8/site-packages'
    if user_install_path in sys.path:
        sys.path.remove(user_install_path)

ignore_user_installs("starzew")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'


def load_pkl(file_name):
    with open(file_name, 'rb') as inp:
        data = dill.load(inp)

    return data


# Globalness g, verticality v, diagonality d

def globalness(head, attention_matrix, n_samples, t):
    # expectation over U -> random word samples
    # T = number of frames = 1500 model.config.max_source_positions  t
    # q stands for query -> A[q,k]
    # (1/T) * sum over q=1 to T [entropy(attention vom head für sample)]
    g = 0
    for i in range(n_samples):
        sum = 0
        for q in range(t):
            sum = np.add(sum, entropy(attention_matrix[i, head, q, :]))

        sum = sum / t
        g += sum

    g = g / n_samples

    return g


def verticality(head, attention_matrix, n_samples, t):
    v = 0

    for i in range(n_samples):
        sum = np.zeros((t))
        for q in range(t):
            sum = np.add(sum, attention_matrix[i, head, q, :])
        sum = sum / t
        v -= entropy(sum)
    v = v / n_samples
    return v


# attention_matrix: [n_samples, n_heads, n_frames, n_frames
def diagonality(head, attention_matrix, n_samples, t):
    d = 0

    for i in range(n_samples):
        sum = 0
        for q in range(t):
            for k in range(t):
                sum += np.abs(q - k) * attention_matrix[i, head, q, k]
        sum = -sum / np.square(t)
        d += sum

    d = d / n_samples

    return d.item()

# calculate head scores for each layer averaged over n_samples samples and save as csv
def head_scores_csv(n_samples, encoder_attentions, n_heads, sequence_len, file_name):
    all_g_scores = []
    all_v_scores = []
    all_d_scores = []
    all_g_scores_b = []
    all_v_scores_b = []
    all_d_scores_b = []

    for layer in range(len(encoder_attentions)):  # go through each layer
        print(f"layer{layer}")
        g_scores = []
        g_scores_b = []
        v_scores = []
        v_scores_b = []
        d_scores = []
        d_scores_b = []
        for head in range(n_heads):
            print(f"head{head}")
            g = globalness(head, encoder_attentions[layer], n_samples, sequence_len)
            g_scores.append(g)
            v = verticality(head, encoder_attentions[layer], n_samples, sequence_len)
            v_scores.append(v)
            d = diagonality(head, encoder_attentions[layer], n_samples, sequence_len)
            d_scores.append(d)

            # baselines
            print("baselines")
            rng = np.random.default_rng()
            ind = np.random.randint(0,encoder_attentions[layer].shape[0])
            b_att = encoder_attentions[layer][ind:ind+1].numpy()
            b_att = rng.permutation(b_att, axis=-1)
            b_att = rng.permutation(b_att, axis=-2)
            #print(encoder_attentions[layer].shape, b_att.shape)

            g = globalness(head, b_att, 1, sequence_len)
            g_scores_b.append(g)
            v = verticality(head, b_att, 1, sequence_len)
            v_scores_b.append(v)
            d = diagonality(head, b_att, 1, sequence_len)
            d_scores_b.append(d)
            #print(g, v, d)

        all_g_scores.append(g_scores)
        all_v_scores.append(v_scores)
        all_d_scores.append(d_scores)
        all_g_scores_b.append(g_scores_b)
        all_v_scores_b.append(v_scores_b)
        all_d_scores_b.append(d_scores_b)


    layers = ["Layer0", "Layer1", "Layer2", "Layer3", "Layer4", "Layer5"]
    heads = ["Head0", "Head1", "Head2", "Head3", "Head4", "Head5", "Head6", "Head7"]
    df = pd.DataFrame(all_g_scores, index=layers, columns=heads)
    df.to_csv(f"head_scores/g_scores_{file_name}.csv")
    df = pd.DataFrame(all_v_scores, index=layers, columns=heads)
    df.to_csv(f"head_scores/v_scores_{file_name}.csv")
    df = pd.DataFrame(all_d_scores, index=layers, columns=heads)
    df.to_csv(f"head_scores/d_scores_{file_name}.csv")

    df = pd.DataFrame(all_g_scores_b, index=layers, columns=heads)
    df.to_csv(f"head_scores/g_scores_{file_name}_b.csv")
    df = pd.DataFrame(all_v_scores_b, index=layers, columns=heads)
    df.to_csv(f"head_scores/v_scores_{file_name}_b.csv")
    df = pd.DataFrame(all_d_scores_b, index=layers, columns=heads)
    df.to_csv(f"head_scores/d_scores_{file_name}_b.csv")


def get_examplary_atts(ds, sev, n_samples, model):
    # get samples
    indices = np.random.randint(len(ds[sev]), size=n_samples)
    subset = [ds[sev][i] for i in indices]

    # predict & get atts
    inputs = feature_extractor(subset, do_normalize=True, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        pred = model(inputs, decoder_input_ids=decoder_input_ids, output_attentions=True)

    return pred.encoder_attentions

def calc_head_scores(dataset, severities, n_samples, model_pt, model_ft):
    print("head scores")
    n_heads = 8
    sequence_len = 1500

    for sev in severities:
        # examplary full attentions without downsampling
        atts_pt = get_examplary_atts(dataset, sev, n_samples, model_pt)
        atts_ft = get_examplary_atts(dataset, sev, n_samples, model_ft)

        file_name = f"{sev}_pt_{n_samples}"
        print(file_name)
        head_scores_csv(n_samples, atts_pt, n_heads, sequence_len, file_name)

        file_name = f"{sev}_ft_{n_samples}"
        print(file_name)
        head_scores_csv(n_samples, atts_ft, n_heads, sequence_len, file_name)


# classify each head for each layer separately into global, vertical, diagonal
def classify_heads(rank, n_heads):
    classifications = []
    classifications_b = []

    for layer in range(len(rank['g'])):
        classification = []  # 0 - global, 1 - vertical, 2 - diagonal
        classification_b = []

        for head in range(n_heads):
            # look for lowest rank
            ranks = [rank['g'].loc[rank['g'].index == f"Layer{layer}", rank['g'].columns == f"Head{head}"],
                     rank['v'].loc[rank['v'].index == f"Layer{layer}", rank['v'].columns == f"Head{head}"],
                     rank['d'].loc[rank['d'].index == f"Layer{layer}", rank['d'].columns == f"Head{head}"]]
            ranks = np.asarray(ranks)
            class_id = np.random.choice(np.flatnonzero(ranks == ranks.min()))  # if tie, random index of winner
            classification.append(class_id)

            # baselines
            ranks_b = [rank['g_b'].loc[rank['g_b'].index == f"Layer{layer}", rank['g_b'].columns == f"Head{head}"],
                     rank['v_b'].loc[rank['v_b'].index == f"Layer{layer}", rank['v_b'].columns == f"Head{head}"],
                     rank['d_b'].loc[rank['d_b'].index == f"Layer{layer}", rank['d_b'].columns == f"Head{head}"]]
            ranks_b = np.asarray(ranks_b)
            class_id_b = np.random.choice(np.flatnonzero(ranks_b == ranks_b.min()))  # if tie, random index of winner
            classification_b.append(class_id_b)

        classifications.append(classification)
        classifications_b.append(classification_b)

    classifications = pd.DataFrame(classifications)
    classifications_b = pd.DataFrame(classifications_b)

    return classifications, classifications_b

def rank_classify(severities, n_samples):
    for variant in ["pt", "ft"]:
        for sev in severities:
            scores = {
                'g':pd.read_csv(f"head_scores/g_scores_{sev}_{variant}_{n_samples}.csv", index_col=0),
                'v':pd.read_csv(f"head_scores/v_scores_{sev}_{variant}_{n_samples}.csv", index_col=0),
                'd':pd.read_csv(f"head_scores/d_scores_{sev}_{variant}_{n_samples}.csv", index_col=0),
                'g_b': pd.read_csv(f"head_scores/g_scores_{sev}_{variant}_{n_samples}_b.csv", index_col=0),
                'v_b': pd.read_csv(f"head_scores/v_scores_{sev}_{variant}_{n_samples}_b.csv", index_col=0),
                'd_b': pd.read_csv(f"head_scores/d_scores_{sev}_{variant}_{n_samples}_b.csv", index_col=0),
            }
            rank = {
                'g': scores['g'].rank(axis=1, ascending=False),
                'v': scores['v'].rank(axis=1, ascending=False),
                'd': scores['d'].rank(axis=1, ascending=False),
                'g_b': scores['g_b'].rank(axis=1, ascending=False),
                'v_b': scores['v_b'].rank(axis=1, ascending=False),
                'd_b': scores['d_b'].rank(axis=1, ascending=False),
            }

            classifications, classifications_b = classify_heads(rank, 8)

            with open(f"head_scores/classifications_{sev}_{variant}_{n_samples}.pkl", "wb") as outp:
                dill.dump(classifications, outp)

            with open(f"head_scores/classifications_{sev}_{variant}_{n_samples}_b.pkl", "wb") as outp:
                dill.dump(classifications_b, outp)


def get_head_masks(classifications, category):  # 0-global, 1-vertical, 2-diagonal
    classifications = np.array(classifications)

    head_mask_og = (classifications == category) * 1
    head_masks = []

    # ids der heads in der cat
    ids = np.argwhere(head_mask_og)

    # mask 0 heads -> mask 1 head -> mask 2 heads ...
    for n_masked_heads in range(len(ids) + 1):
        # select random heads of cat to mask
        ids_masked_heads = np.random.choice(len(ids), size=n_masked_heads, replace=False)
        # print(ids_masked_heads)
        ids_masked_heads = [ids[i] for i in ids_masked_heads]

        # mask heads
        head_mask = np.ones_like(head_mask_og)
        for i in ids_masked_heads:
            head_mask[i[0], i[1]] = 0

        head_mask = torch.from_numpy(head_mask)
        head_masks.append(head_mask)

    return head_masks


def get_head_masks_layer(classifications, category, layer):  # 0-global, 1-vertical, 2-diagonal
    classifications = np.array(classifications)

    head_mask_og = (classifications == category) * 1
    head_masks = []

    # ids der heads in der cat
    ids = np.argwhere(head_mask_og[layer, :])
    # print(ids)

    # mask 0 heads -> mask 1 head -> mask 2 heads ... all possible combinations
    for n_masked_heads in range(len(ids) + 1):
        for comb in combinations(ids, n_masked_heads):

            # mask heads
            head_mask = np.ones_like(head_mask_og)
            for i in comb:
                head_mask[layer, i[0]] = 0

            head_mask = torch.from_numpy(head_mask)
            head_masks.append(head_mask)

    return head_masks


def get_head_masks_baseline():  # 0-global, 1-vertical, 2-diagonal
    head_masks = []
    head_mask_og = np.ones((6, 8))

    ids = []
    for i in range(6):
        for j in range(8):
            ids.append([i, j])

    # mask 0 heads -> mask 1 head -> mask 2 heads ...
    for n_masked_heads in range(len(ids) + 1):
        # select random heads of cat to mask
        ids_masked_heads = np.random.choice(len(ids), size=n_masked_heads, replace=False)
        # print(ids_masked_heads)
        ids_masked_heads = [ids[i] for i in ids_masked_heads]

        # mask heads
        head_mask = np.ones_like(head_mask_og)
        for i in ids_masked_heads:
            head_mask[i[0], i[1]] = 0

        head_mask = torch.from_numpy(head_mask)
        head_masks.append(head_mask)

    return head_masks

def head_masks(version,ptft):# version z.B. m_pt_2
    classifications = load_pkl(f'head_scores/classifications_{version}.pkl')
    categories = [0,1,2]
    cat_names = ["global","vertical","diagonal"]
    for i,cat in enumerate(categories):
        for j in range(10): # 10 x random overall masks
            head_masks = get_head_masks(classifications, cat)
            # overall random head masks
            with open(f"head_scores/masks_{cat_names[i]}_total_{j}_{ptft}.pkl", 'wb') as outp:
                dill.dump(head_masks, outp)
        # layerwise head masks
        for layer in range(6):
            head_masks = get_head_masks_layer(classifications, cat, layer)
            with open(f"head_scores/masks_{cat_names[i]}_{layer}_{ptft}.pkl", 'wb') as outp:
                dill.dump(head_masks, outp)

    head_masks_baseline = get_head_masks_baseline()
    with open("head_scores/masks_baseline.pkl", 'wb') as outp:
        dill.dump(head_masks_baseline, outp)

# task: "asr_ua.py"/"sev_ua.py"/"sid_ua.py"
def head_ablation(task, expname, ptft, gvd):


    # do experiment for each layer
    print(task, expname, gvd, "layerwise")
    results_layerwise = dict()
    for layer in range(6):
        print("layer", layer)
        results_layerwise[layer]  = []
        masks = load_pkl(f'head_scores/masks_{gvd}_{layer}_{ptft}.pkl')

        for mask in masks:
            with open("head_scores/mask_to_use.pkl", 'wb') as outp:
                dill.dump(mask, outp)
            target_dir =f"result/{expname}"
            start = 3
            if task == ("asr_ua.py"):
                start = 4
            args = ["python3", task, "--target_dir", target_dir, "--prepare_data.dataset_root", "datasets", "--build_upstream.name", "whisper_pt", "--start", f"{start}"]
            #print("targetdir", target_dir)
            subprocess.run(args, stdout = subprocess.DEVNULL,stderr=subprocess.DEVNULL) #,stderr=subprocess.DEVNULL

            # load results
            if task == "pr_ua.py":
                with open(f"/project/thesis/result/{expname}/evaluate/valid_best/test_asr/result.yaml") as f:
                    result = yaml.load(f, Loader=yaml.FullLoader)
                    results_layerwise[layer].append(result)
            else:
                with open(f"/project/thesis/result/{expname}/evaluate/valid_best/test_{task[:3]}/result.yaml") as f:
                    result = yaml.load(f, Loader=yaml.FullLoader)
                    results_layerwise[layer].append(result)

    with open(f"head_scores/ablation_{expname}_{gvd}_layerwise_{ptft}.pkl", 'wb') as outp:
        dill.dump(results_layerwise, outp)

    print(task, expname, gvd, "total")
    # do overall experiments
    results_subsets = dict()
    for i in range(10):
        print("subset",i)
        results_subsets[i] = []
        masks = load_pkl(f'head_scores/masks_{gvd}_total_{i}_{ptft}.pkl')
        for mask in masks:
            with open("head_scores/mask_to_use.pkl", 'wb') as outp:
                dill.dump(mask, outp)
            target_dir = f"result/{expname}"
            print("target_dir", target_dir)
            start = 3
            if task == ("asr_ua.py"):
                start = 4
            args = ["python3", task, "--target_dir", target_dir, "--prepare_data.dataset_root", "datasets",
                    "--build_upstream.name", "whisper_pt", "--start", f"{start}"]
            subprocess.run(args, stdout = subprocess.DEVNULL,stderr=subprocess.DEVNULL)

            # load results
            if task == "pr_ua.py":
                with open(f"/project/thesis/result/{expname}/evaluate/valid_best/test_asr/result.yaml") as f:
                    result = yaml.load(f, Loader=yaml.FullLoader)
                    results_subsets[i].append(result)
            else:
                with open(f"/project/thesis/result/{expname}/evaluate/valid_best/test_{task[:3]}/result.yaml") as f:
                    result = yaml.load(f, Loader=yaml.FullLoader)
                    results_subsets[i].append(result)

    # save results
    with open(f"head_scores/ablation_{expname}_{gvd}_total_{ptft}.pkl", 'wb') as outp:
        dill.dump(results_subsets, outp)




    '''
    for i in range(len(masks)):
        test_name = "test-clean"
        model = f"whisper_pt_masked{i}"
        ckpt = "/project/thesis/s3prl/s3prl/result/downstream/exp1/dev-clean-best.ckpt"
        expdir =  "pt_asr_test1"
        args = ["python3", "run_downstream.py", "-m", "evaluate", "-t", test_name, "-i", ckpt, "-u", model, "-d", task, "-n", expdir ]
        result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
        print(result.stdout)
        return
     '''


    ####################################################


if __name__ == "__main__":
    ua_sampling_rate = 16000

    print("load model")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", attn_implementation="eager")
    model_ft = WhisperForConditionalGeneration.from_pretrained("/project/thesis/model/whisper-base-ft/checkpoint-21", attn_implementation="eager")

    for m in [model, model_ft]:
        m.generation_config.language = "english"
        m.generation_config.task = "transcribe"
        m.generation_config.is_multilingual = False
        m.generation_config.forced_decoder_ids = None

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    # Calculate g-,v-, d-scores
    #severities = ["v_l", "l","m", "h", "all", "c"]
    #n_samples = 10
    #ds = load_pkl('datasets/final.pkl')
    #calc_head_scores(ds, severities, n_samples, model, model_ft)
    #rank_classify(severities,n_samples)

    #version = "all_pt_10"
    #head_masks(version,"pt")
    #version = "all_ft_10"
    #head_masks(version, "ft")

    #tasks = ["sev_ua.py","sid_ua.py","asr_ua.py"]
    tasks = ["pr_ua.py"]
    #expnames = ["exp_sev_pt", "exp_sid_pt", "exp_asr_pt", "exp_sev_ft", "exp_sid_ft", "exp_asr_ft"]
    expnames = ["exp_pr_pt", "exp_pr_ft"]
    for task, expname in zip(tasks+tasks, expnames):
        head_ablation(task, expname, expname[-2:],"global")
        head_ablation(task, expname, expname[-2:],"vertical")
        head_ablation(task, expname, expname[-2:],"diagonal")

    #print("finished")
