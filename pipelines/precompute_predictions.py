import os
import sys
import torch
import torchvision.transforms as T
import numpy as np
from transformers import  WhisperForConditionalGeneration, AutoFeatureExtractor
import dill


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

# get time-averaged representations of hidden_states of all given samples
def get_tavg(encoder_hidden_states):
    avgs = []
    for hidden_state in encoder_hidden_states:
        avgs.append(torch.mean(hidden_state, dim=1)) # average representation over time-axis: [n_samples,1500,512] -> [n_samples,512]

    return avgs

def get_favg(encoder_hs, num_frames):
    avgs = []
    for hs in encoder_hs: # shape = [303,1500,512]
        avg = []
        # go through each sample
        for i in range(hs.shape[0]):
            representation = hs[i,:int((num_frames[i]-1)/2),:]
            # avg over frames corresponding to sample
            representation = torch.mean(representation, dim=0)
            avg.append(np.array(representation))
        avg = np.array(avg)
        avgs.append(avg)

    return avgs


def get_cumatts(atts):
    layer_ca = []
    for layer in range(6):
        cumatts = []
        for k in range(atts[layer].shape[-1]):
            cumatt = torch.sum(atts[layer][:, :, :, k], dim=-1)
            # average over heads
            cumatt = torch.mean(cumatt, dim=-1)
            cumatts.append(np.array(cumatt))
        layer_ca.append(cumatts)
    return layer_ca

#https://scipython.com/blog/binning-a-2d-array-in-numpy/
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def layerwise_rebin(arrs, new_shape):
    return np.array(list(map(lambda x: rebin(x, new_shape), arrs)))

def att_representation(atts, variant, dim=None):
    # atts: 6x [10,8,1500,1500]
    # dim e.g. (30,30)
    # avg attentions across heads
    atts = list(map(lambda x: np.mean(np.array(x), axis=1), atts))  # 6x [n_samples,1500,1500]
    n_samples=atts[0].shape[0]

    # create representation
    if variant == "rebin":
        atts = np.array(list(map(lambda x: layerwise_rebin(x, (dim, dim)), atts)))  # [n_samples,dim,dim]
        atts = atts.reshape([6, n_samples, dim * dim])  # [6, n_samples, dim*dim]
    elif variant == "avgpool":
        atts = np.array(atts)  # [6, n_samples, 1500, 1500]
        atts = torch.from_numpy(atts)
        atts = torch.nn.AdaptiveAvgPool2d((dim,dim))(atts)
        atts = np.array(atts)
        atts = atts.reshape([6, n_samples, atts.shape[-1] * atts.shape[-1]])  # [6, n_samples, dim?*dim?]
    elif variant == "resize":
        atts = np.array(atts)  # [6, n_samples, 1500, 1500]
        atts = torch.from_numpy(atts)
        atts = T.functional.resize(atts, size=[dim, dim])
        atts = np.array(atts)
        atts = atts.reshape([6, n_samples, dim * dim])  # [6, n_samples, dim*dim]
    elif variant == "basic":
        atts = np.array(atts)  # [6, n_samples, 1500, 1500]
        atts = atts.reshape([6, n_samples, 1500 * 1500])
        atts = np.array(atts)
    else:
        raise ValueError('Wrong variant')

    return atts

# downsample attentions
def get_datt(atts):
    # atts 6x [batch_size, num_heads, sequence_length, sequence_length]
    variant = "avgpool"
    dim = 22
    datts = att_representation(atts, variant, dim=dim)
    return datts

# tavgs get stored in subsets that have to be merged correctly n_subsets x n_layers x [n_samples, 512] -> n_layers x [n_samples_all, 512]
def merge_avg(file_name, tavg_or_favg, blocks):
    file_names = [f'/{tavg_or_favg}/{file_name}.pkl']
    if blocks:
        for block in ["B1", "B2", "B3"]:
            file_names.append(f'/{tavg_or_favg}/{file_name}_{block}.pkl')


    for name in file_names:
        #print(name)
        data = []
        with open(name, 'rb') as inp:
            try:
                while True:  # load all subsets of the file
                    data.append(dill.load(inp))
            except EOFError:
                pass

        # merge all subsets together -> return
        merged = [[], [], [], [], [], [], []]
        for subset in data:
            for i, layer in enumerate(subset):
                try:
                    if np.array(subset).shape[-1] == 0:  # no B1 in this subset
                        continue
                except ValueError:
                    pass
                #print(np.array(layer).shape)
                merged[i].append(layer)

        for i, layer in enumerate(merged):
            merged[i] = np.concatenate(layer)

        with open(name, 'wb') as outp:
            dill.dump(np.array(merged), outp)

def merge_atts(file_name, blocks):
    file_names = [f'/attentions/{file_name}.pkl']
    if blocks:
        for block in ["B1", "B2", "B3"]:
            file_names.append(f'/attentions/{file_name}_{block}.pkl')

    for name in file_names:
        data = []
        with open(f'/attentions/{file_name}.pkl', 'rb') as inp:
            try:
                while True:  # load all subsets of the file
                    data.append(dill.load(inp))
            except EOFError:
                pass

        atts = np.concatenate(data, axis=1)

        with open(name, "wb") as outp:
            dill.dump(atts, outp)

def merge_cumatts(file_name, blocks):
    file_names = [f'/cumatt/{file_name}.pkl']
    if blocks:
        for block in ["B1", "B2", "B3"]:
            file_names.append(f'/cumatt/{file_name}_{block}.pkl')

    for name in file_names:
        data = []
        with open(f'/cumatt/{file_name}.pkl', 'rb') as inp:
            try:
                while True:  # load all subsets of the file
                    data.append(dill.load(inp))
            except EOFError:
                pass

        cumatts = np.concatenate(data, axis=-1)

        with open(name, "wb") as outp:
            dill.dump(cumatts, outp)


def split_blocks(infos, block_ids, tavg_favg, file_name):
    block1 = []
    block2 = []
    block3 = []

    #print("range len", len(infos))
    for layer in range(len(infos)):
        #print(infos[layer].shape, tavg_favg)
        block1.append([np.array(infos[layer][i]) for i, id in enumerate(block_ids) if id=="B1"])
        block2.append([np.array(infos[layer][i]) for i, id in enumerate(block_ids) if id == "B2"])
        block3.append([np.array(infos[layer][i]) for i, id in enumerate(block_ids) if id == "B3"])
        #print(len(block1[layer]), len(block2[layer]), len(block3[layer]), len(block_ids))

    names = ["B1", "B2", "B3"]
    for i,block in enumerate([block1,block2,block3]):
        with open(f"/{tavg_favg}/{file_name}_{names[i]}.pkl", "ab") as outp:
            dill.dump(np.array(block), outp)




def predict_inputs(inputs, model, file_name, save_representations, save_favgs, save_attentions, block_ids):
    if save_favgs:
        # if we want to calc favgs too, we need num_frames
        num_frames = inputs.num_frames
    else:
        num_frames = None
    inputs = inputs.input_features

    with torch.no_grad():
        print("predict")
        n_samples = len(inputs)
        for i in range(1,11): # predict 10 subsets of samples
            print(i)
            #continue # REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ind_before= int((i-1)*(n_samples/10))
            ind = int(i*(n_samples/10))
            pred = model(inputs[ind_before:ind], decoder_input_ids=decoder_input_ids,
                              output_hidden_states=(save_representations or save_favgs), output_attentions=save_attentions)

            if save_representations:
                #print("tavg")
                tavg = get_tavg(pred.encoder_hidden_states)
                split_blocks(tavg, block_ids[ind_before:ind], "tavg", file_name)
                with open(f"/tavg/{file_name}.pkl", "ab") as outp:
                    dill.dump(tavg,outp)
                del tavg

            if save_favgs:
                #print("favg")
                favg = get_favg(pred.encoder_hidden_states, num_frames[ind_before:ind])
                split_blocks(favg, block_ids[ind_before:ind], "favg", file_name)
                with open(f"/favg/{file_name}.pkl", "ab") as outp:
                    dill.dump(favg,outp)
                del favg

            if save_attentions:
                #print("attention")
                atts = get_datt(pred.encoder_attentions)
                split_blocks(atts, block_ids[ind_before:ind], "attentions", file_name)
                with open(f'/attentions/{file_name}.pkl', 'ab') as outp:
                    dill.dump(atts, outp)

                cumatts = get_cumatts(pred.encoder_attentions)
                split_blocks(cumatts, block_ids[ind_before:ind], "cumatt", file_name)
                with open(f'/cumatt/{file_name}.pkl', 'ab') as outp:
                    dill.dump(cumatts, outp)

            del pred
        if save_representations:
            merge_avg(file_name, "tavg", blocks=True)

        if save_favgs:
            merge_avg(file_name, "favg", blocks=True)

        if save_attentions:
            merge_atts(file_name, blocks=True)
            merge_cumatts(file_name, blocks=True)


def merge_info_all(tavg_favg_attentions_cumatt, file, version, axis, blocks):
    info = dict()
    sevs = ["v_l", "l", "m", "h"]
    for sev in sevs:
        info[sev] = load_pkl(f'/{tavg_favg_attentions_cumatt}/{file}_{sev}_{version}.pkl')

    infos_all = np.concatenate([info[sev] for sev in sevs], axis=axis)
    with open(f"/{tavg_favg_attentions_cumatt}/{file}_all_{version}.pkl", "wb") as outp:
        dill.dump(infos_all, outp)

    if blocks:
        for block in ["B1","B2","B3"]:
            info = dict()
            sevs = ["v_l", "l", "m", "h"]
            for sev in sevs:
                info[sev] = load_pkl(f'{tavg_favg_attentions_cumatt}/{file}_{sev}_{version}_{block}.pkl')

            infos_all = np.concatenate([info[sev] for sev in sevs], axis=axis)
            with open(f"{tavg_favg_attentions_cumatt}/{file}_all_{version}_{block}.pkl", "wb") as outp:
                dill.dump(infos_all, outp)

def merge_sev_all(file, save_tavgs, save_favgs, save_attentions):

    for version in ["pt", "ft"]:
        if save_tavgs:
            merge_info_all("tavg", file, version, axis=1, blocks=True)
        if save_favgs:
            merge_info_all("favg", file, version, axis=1, blocks=True)
        if save_attentions:
            merge_info_all("attentions", file, version, axis=1, blocks=True)
            merge_info_all("cumatt", file, version, axis=-1, blocks=True)



def save_tavgs_attentions(file_names, subsets, model_pt, model_ft, save_representations, save_favgs,  save_attentions):
    for file in file_names:
        ds = load_pkl(f"/datasets/{file}.pkl")
        print(file)
        for subset in subsets:
            print(subset)
            inputs = feature_extractor(ds[subset], return_token_timestamps=save_favgs, do_normalize=True, sampling_rate=16000,
                                       return_tensors="pt")
            print("pt")
            predict_inputs(inputs, model_pt, f"{file}_{subset}_pt", save_representations, save_favgs, save_attentions, ds[f"block_id_{subset}"])
            print("ft")
            predict_inputs(inputs, model_ft, f"{file}_{subset}_ft", save_representations, save_favgs, save_attentions, ds[f"block_id_{subset}"])
            del inputs
        del ds

        merge_sev_all(file, save_tavgs, save_favgs, save_attentions)



####################################################


if __name__ == "__main__":
    ua_sampling_rate = 16000

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
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    # save tavgs and attentions
    file_names = ["final"]
    subsets = ["v_l", "l", "m", "h", "c"]
    save_tavgs= True
    save_attentions = True
    save_favgs = True
    save_tavgs_attentions(file_names, subsets, model, model_ft, save_tavgs, save_favgs, save_attentions)



    print("finished")

