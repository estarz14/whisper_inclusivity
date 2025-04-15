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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HTTP_PROXY'] = 'http://fp.cs.ovgu.de:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp.cs.ovgu.de:3210/'

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration, AutoProcessor, WavLMModel, AutoFeatureExtractor
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.patches as mpatches
import dill
from torch.nn.functional import cosine_similarity
import IPython
from sklearn.metrics import accuracy_score
import itertools
from sklearn import preprocessing

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")


def load_pkl(file_name):
    with open(file_name, 'rb') as inp:
        data = dill.load(inp)

    return data


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def labelencode(ds_severities, severity):
    le = preprocessing.LabelEncoder()
    ds_severities[f"speakers_{severity}"] = le.fit_transform(ds_severities[f"speakers_{severity}"])
    ds_severities[f"transcripts_{severity}"] = le.fit_transform(ds_severities[f"transcripts_{severity}"])

def labelencode_sev(ds_severities):
    le = preprocessing.LabelEncoder()
    ds_severities["severities_all"] = le.fit_transform(ds_severities["severities_all"])


def create_dl(severity, split, batch_size,sr):
    # extract features -------> CAUTION: HERE ONLY 500 SAMPLES AT THE MOMENT (if [:500])
    inputs = feature_extractor(ds_severities[severity][:500], do_normalize=True, sampling_rate=sr, return_tensors="pt")

    # Create datasets
    ds_speakers = CustomDataset(inputs.input_features, ds_severities[f"speakers_{severity}"])
    ds_words = CustomDataset(inputs.input_features, ds_severities[f"transcripts_{severity}"])

    # train/test split
    train_speakers, test_speakers = torch.utils.data.random_split(ds_speakers, split)
    train_words, test_words = torch.utils.data.random_split(ds_words, split)

    # Create data loader for train and test data
    dataloaders = {
        "train_speakers": DataLoader(train_speakers, batch_size=batch_size, shuffle=True, drop_last=True),
        "test_speakers": DataLoader(test_speakers, batch_size=batch_size, shuffle=False, drop_last=True),
        "train_words": DataLoader(train_words, batch_size=batch_size, shuffle=True, drop_last=True),
        "test_words": DataLoader(test_words, batch_size=batch_size, shuffle=False, drop_last=True),
    }

    return dataloaders


def create_dl_sev(split, batch_size, sr):
    # extract features -------> CAUTION: HERE ONLY 500 SAMPLES AT THE MOMENT (if [:500])
    inputs = feature_extractor(ds_severities["all"], do_normalize=True, sampling_rate=sr, return_tensors="pt")

    # create datasets
    ds = CustomDataset(inputs.input_features, ds_severities["severities_all"])

    # train/test split
    train, test = torch.utils.data.random_split(ds, split)

    # Data loaders for train and test data
    dataloaders = {
        "train_severities": DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        "test_severities": DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    }

    return dataloaders


class Probe(nn.Module):
    def __init__(self, base_model, hidden_dim, n_classes, batch_size):
        super().__init__()
        self.encoder = base_model.encoder
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(7)])
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, x, batch_size):
        x = self.encoder(x, output_hidden_states=True)
        weighted_avg = x.hidden_states[0] * self.weights[0]
        for i in range(1, 7):
            weighted_avg += x.hidden_states[i] * self.weights[i]
        x = torch.mean(weighted_avg, dim=1)  # avg over time frames -> Yang et al. 2021 Mean Pool
        x = self.head(x.reshape((batch_size, 512)))

        return x


def freeze_enc(probe):
    # Freeze Whisper encoder part of probing model
    for param in probe.encoder.parameters():
        param.requires_grad = False


def create_probes(model, model_ft, n_speakers, n_words, batch_size, hidden_dim, n_severities=None):
    if n_severities:
        probe_pt = Probe(model, hidden_dim, n_severities, batch_size)
        probe_ft = Probe(model_ft, hidden_dim, n_severities, batch_size)

        probes = {
            "severities_pt": probe_pt,
            "severities_ft": probe_ft
        }
    else:
        probe_speakers_pt = Probe(model, hidden_dim, n_speakers, batch_size)
        probe_word_pt = Probe(model, hidden_dim, n_words, batch_size)
        probe_speakers_ft = Probe(model_ft, hidden_dim, n_speakers, batch_size)
        probe_word_ft = Probe(model_ft, hidden_dim, n_words, batch_size)

        probes = {
            "speakers_pt": probe_speakers_pt,
            "speakers_ft": probe_speakers_ft,
            "words_pt": probe_word_pt,
            "words_ft": probe_word_ft,
        }

    # freeze whisper encoders
    for _, probe in probes.items():
        freeze_enc(probe)

    return probes


def create_optimizers(probes, lr, severities=False):
    if severities:
        optimizers = {
            "severities_pt": torch.optim.Adam(probes["severities_pt"].parameters(), lr=lr),
            "severities_ft": torch.optim.Adam(probes["severities_ft"].parameters(), lr=lr),
        }
    else:
        optimizers = {
            "speakers_pt": torch.optim.Adam(probes["speakers_pt"].parameters(), lr=lr),
            "speakers_ft": torch.optim.Adam(probes["speakers_ft"].parameters(), lr=lr),
            "words_pt": torch.optim.Adam(probes["words_pt"].parameters(), lr=lr),
            "words_ft": torch.optim.Adam(probes["words_ft"].parameters(), lr=lr),
        }

    return optimizers


def eval_probe(probe, dataloader, batch_size):
    # Evaluate the probe
    preds = []
    labels = []

    for i, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            pred = probe(inputs, batch_size)
        pred = np.argmax(pred, axis=1)
        preds.append(np.array(pred))
        labels.append(np.array(targets))

    preds = np.ravel(preds)
    labels = np.ravel(labels)
    acc = accuracy_score(preds, labels)

    return acc


def train_probe(epochs, probe, dataloader_train, dataloader_test, optimizer, loss_fn, batch_size):
    accs = []
    weights = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        for i, (inputs, targets) in enumerate(dataloader_train):  # go through batches
            optimizer.zero_grad()  # clear gradients
            pred = probe(inputs, batch_size)
            loss = loss_fn(pred, torch.as_tensor(targets))
            loss.backward()
            optimizer.step()  # update model weights

        accs.append(eval_probe(probe, dataloader_test, batch_size))
        # store weights as floats not params
        weight = []
        for w in probe.weights:
            weight.append(w.detach().numpy()[0])
        weights.append(weight)

    history = {
        'accs': accs,
        'weights': weights
    }
    return history


def training(probes, dl, optimizers, loss_fn, epochs, batch_size, severities=False):
    if severities:
        history = {
            "severities_pt": train_probe(epochs, probes["severities_pt"], dl["train_severities"], dl["test_severities"],
                                         optimizers["severities_pt"], loss_fn, batch_size),
            "severities_ft": train_probe(epochs, probes["severities_ft"], dl["train_severities"], dl["test_severities"],
                                         optimizers["severities_ft"], loss_fn, batch_size),
        }
    else:
        history = {
            "speakers_pt": train_probe(epochs, probes["speakers_pt"], dl["train_speakers"], dl["test_speakers"],
                                       optimizers["speakers_pt"], loss_fn, batch_size),
            "speakers_ft": train_probe(epochs, probes["speakers_ft"], dl["train_speakers"], dl["test_speakers"],
                                       optimizers["speakers_ft"], loss_fn, batch_size),
            "words_pt": train_probe(epochs, probes["words_pt"], dl["train_words"], dl["test_words"],
                                    optimizers["words_pt"], loss_fn, batch_size),
            "words_ft": train_probe(epochs, probes["words_ft"], dl["train_words"], dl["test_words"],
                                    optimizers["words_ft"], loss_fn, batch_size),
        }

    return history

def speaker_word(ds_severities):
    severities = ["v_l", "l", "m", "h", "all"]
    #severities = ["m"]
    split = [0.8, 0.2]
    hidden_dim = 1024
    batch_size = 32
    loss_fn = nn.CrossEntropyLoss()
    lr = 1e-4  # how to set correctly?
    epochs = 10
    histories = dict()
    sr = 16000

    for severity in severities:
        print(f"Severity: {severity}")
        # transform labels from strings to numbers
        print("Label Encoding")
        labelencode(ds_severities, severity)

        # get n_speakers, n_words
        n_speakers = ds_severities[f"speakers_{severity}"].max() + 1
        n_words = ds_severities[f"transcripts_{severity}"].max() + 1

        # create dataloaders
        print("Data Loaders")
        dl = create_dl(severity, split, batch_size, sr)

        # create probes
        print("Probes")
        probes = create_probes(model, model_ft, n_speakers, n_words, batch_size, hidden_dim)

        # create optimizers
        print("Optimizers")
        optimizers = create_optimizers(probes, lr)

        # train probes & store history
        print("Training")
        histories[severity] = training(probes, dl, optimizers, loss_fn, epochs, batch_size)
    return histories

ds_severities = load_pkl("datasets/severities.pkl")
labelencode_sev(ds_severities)


def sev_pipeline(split, batch_size, sr, n_severities, loss_fn, epochs, lr):
    # create dataloaders
    print("Data Loaders")
    dl = create_dl_sev(split, batch_size, sr)

    # create probes
    print("Probes")
    probes = create_probes(model, model_ft, None, None, batch_size, n_severities)

    # create optimizers
    print("Optimizers")
    optimizers = create_optimizers(probes, lr, severities=True)

    # train probes & store history
    print("Training")
    histories = training(probes, dl, optimizers, loss_fn, epochs, batch_size)

    with open('histories/sevs.pkl', 'ab') as outp:
        dill.dump(histories, outp)

def probe_severity(ds_severities):
    split = [0.8, 0.2]
    hidden_dim = 1024
    batch_size = 32
    sr = 16000
    loss_fn = nn.CrossEntropyLoss()
    lr = 1e-4  # how to set correctly?
    epochs = 10
    histories2 = dict()

    n_severities = len(["v_l", "l", "m", "h"])

    sev_pipeline(split, batch_size, sr, n_severities, loss_fn, epochs, lr)



    return histories

# Load model & Feature Extractor
model = WhisperModel.from_pretrained("openai/whisper-base")
model_ft = WhisperModel.from_pretrained("hiwden00/dysarthria-base")

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id


# probe for speaker and word
print("probe speakers and words")
histories = speaker_word(ds_severities)
with open('histories/speakers_words_500.pkl', 'wb') as outp:
    dill.dump(histories, outp)
    
    
# probe for severities
#print("probe sevs")
#histories = probe_severity(ds_severities)
#with open('histories/severities.pkl', 'wb') as outp:
#    dill.dump(histories, outp)

