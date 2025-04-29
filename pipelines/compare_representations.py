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
import numpy as np
import random
import dill
from torch.nn.functional import cosine_similarity
import sklearn
from cka import cca_core
from cka import CKA
from cka import pwcca as _pwcca

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")

def load_pkl(file_name):
    with open(file_name, 'rb') as inp:
        data = dill.load(inp)

    return data


def svcca(acts1, acts2, b1, b2):
    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts1 - np.mean(acts2, axis=1, keepdims=True)
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)
    svacts1 = np.dot(s1 * np.eye(s1.shape[0]), V1)
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2 * np.eye(s2.shape[0]), V2)
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-6, verbose=False)
    # Mean subtract baseline activations
    cb1 = b1 - np.mean(b1, axis=0, keepdims=True)
    cb2 = b2 - np.mean(b2, axis=0, keepdims=True)
    # Perform SVD
    Ub1, sb1, Vb1 = np.linalg.svd(cb1, full_matrices=False)
    Ub2, sb2, Vb2 = np.linalg.svd(cb2, full_matrices=False)
    svb1 = np.dot(sb1 * np.eye(sb1.shape[0]), Vb1)
    svb2 = np.dot(sb2 * np.eye(sb2.shape[0]), Vb2)
    svcca_baseline = cca_core.get_cca_similarity(svb1, svb2, epsilon=1e-6, verbose=False)
    return svcca_results, svcca_baseline



def cosine_matrix(embeds1, embeds2, ids, b1=None, b2=None):
    # embeds shape layer x [n_samples, hidden_dim]
    n = len(embeds1)
    result = np.zeros((n, n))
    result_baseline = np.zeros((n, n))

    for layer1 in range(n):
        for layer2 in range(n):
            embeds1layer1 = [embeds1[layer1][i] for i in ids]
            embeds1layer1 = np.array(embeds1layer1)
            embeds2layer2 = embeds2[layer2]

            embeds_copy1 = embeds1layer1.copy()
            embeds_copy2 = embeds2layer2.copy()
            try:
                cosine = cosine_similarity(embeds1layer1, embeds2layer2)
                result[layer1,layer2] = torch.mean(cosine)

                cosine_b = cosine_similarity(embeds_copy1, embeds_copy2)
                result_baseline[layer1,layer2] = torch.mean(cosine_b)
            except:
                cosine = sklearn.metrics.pairwise.cosine_similarity(embeds1layer1, embeds2layer2)
                result[layer1,layer2] = np.mean(cosine)

                cosine_b = sklearn.metrics.pairwise.cosine_similarity(embeds_copy1, embeds_copy1)
                result_baseline[layer1,layer2] = np.mean(cosine_b)

    cosines_dict = {
            "results": result,
            "baseline": result_baseline
        }
    return cosines_dict


def pwcca_matrix(embeds1, embeds2, ids):
    # embeds shape layer x [n_samples, hidden_dim]
    n = len(embeds1)
    result = np.zeros((n, n))
    result_baseline = np.zeros((n, n))


    for layer1 in range(n):
        for layer2 in range(n):
            embeds1layer1 = [embeds1[layer1][i] for i in ids]
            embeds1layer1 = np.array(embeds1layer1)
            embeds2layer2 = embeds2[layer2]

            result[layer1, layer2], _, _ = _pwcca.compute_pwcca(embeds1layer1.transpose(),
                                           embeds2layer2.transpose(), epsilon=1e-6)
            
            embeds_copy1 = embeds1layer1.copy()
            embeds_copy2 = embeds2layer2.copy()
            result_baseline[layer1, layer2], _, _ = _pwcca.compute_pwcca(embeds_copy1.transpose(), embeds_copy1.transpose(), epsilon=0.001)

    pwccas_dict = {
        "results": result,
        "baseline": result_baseline,
    }

    return pwccas_dict


def check_num_samples(embeds1, embeds2, layer1, layer2):
    # sometimes some severities have more examples for a word than others -> make sure that amount of examples is same
    embeds1layer1 = embeds1[layer1].copy()
    embeds2layer2 = embeds2[layer2].copy()

    if embeds1[layer1].shape[0] < embeds2[layer2].shape[0]:
        inds = np.random.randint(0,embeds2[layer2].shape[0], size=embeds1[layer1].shape[0])
        embeds2layer2 = [embeds2[layer2][i] for i in inds]
        embeds2layer2 = np.array(embeds2layer2)
    elif embeds1[layer1].shape[0] > embeds2[layer2].shape[0]:
        inds = np.random.randint(0, embeds1[layer1].shape[0], size=embeds2[layer2].shape[0])
        embeds1layer1 = [embeds1[layer1][i] for i in inds]
        embeds1layer1 = np.array(embeds1layer1)


    return embeds1layer1, embeds2layer2

def svcca_matrix(embeds1, embeds2, ids):
    # embeds shape layer x [n_samples, hidden_dim]
    nlen = len(embeds1)
    result = np.zeros((nlen, nlen))
    result_baseline = np.zeros((nlen,nlen))

    for layer1 in range(nlen):
        for layer2 in range(nlen):
            # sometimes some severities have more examples for a word than others -> make sure that amount of examples is same
            embeds1layer1 = [embeds1[layer1][i] for i in ids]
            embeds1layer1 = np.array(embeds1layer1)
            embeds2layer2 = embeds2[layer2]

            embeds_copy1 = embeds1layer1.copy()
            embeds_copy2 = embeds2layer2.copy()

            svcca_result, svcca_baseline = svcca(embeds1layer1.transpose(),embeds2layer2.transpose(), embeds_copy1.transpose(), embeds_copy1.transpose())
            result[layer1,layer2] = np.mean(svcca_result["cca_coef1"])
            result_baseline[layer1,layer2] = np.mean(svcca_baseline["cca_coef1"])

    svccas_dict = {
        "results": result,
        "baseline": result_baseline,
    }

    return svccas_dict

def cka_layerwise(embeds1, embeds2):
    # embeds shape layer x [n_samples, hidden_dim]
    n = len(embeds1)
    results = []

    for layer in range(n):
        # sometimes some severities have more examples for a word than others -> make sure that amount of examples is same
        if embeds1[layer].shape[0] < embeds2[layer].shape[0]:
            embeds2[layer] = embeds2[layer][:embeds1[layer].shape[0]]
        elif embeds1[layer].shape[0] > embeds2[layer].shape[0]:
            embeds1[layer] = embeds1[layer][:embeds2[layer].shape[0]]
        # calculate CKA
        results.append(CKA.linear_CKA(embeds1[layer], embeds2[layer]))

    return results

def cka_matrix(embeds1, embeds2, ids):
    # embeds shape layer x [n_samples, hidden_dim] -> cka is not calculated for single examples
    n = len(embeds1)
    result = np.zeros((n, n))
    result_baseline = np.zeros((n,n))

    for layer1 in range(n):
        for layer2 in range(n):
            # sometimes some severities have more examples for a word than others -> make sure that amount of examples is same
            embeds1layer1 = [embeds1[layer1][i] for i in ids]
            embeds1layer1 = np.array(embeds1layer1)
            embeds2layer2 = embeds2[layer2]

            result[layer1, layer2] = CKA.linear_CKA(embeds1layer1, embeds2layer2)

            embeds_copy1 = embeds1layer1.copy()
            embeds_copy2 = embeds2layer2.copy()

            result_baseline[layer1, layer2] = CKA.linear_CKA(embeds_copy1, embeds_copy1)

    return result, result_baseline

def compare_to_control(tavg_favg_att, pt_ft):
    ds = load_pkl('/datasets/final.pkl')

    infos = {
        'all': load_pkl(f'/{tavg_favg_att}/final_all_{pt_ft}.pkl'),
        'v_l': load_pkl(f'/{tavg_favg_att}/final_v_l_{pt_ft}.pkl'),
        'l': load_pkl(f'/{tavg_favg_att}/final_l_{pt_ft}.pkl'),
        'm': load_pkl(f'/{tavg_favg_att}/final_m_{pt_ft}.pkl'),
        'h': load_pkl(f'/{tavg_favg_att}/final_h_{pt_ft}.pkl'),
        'c': load_pkl(f'/{tavg_favg_att}/final_c_{pt_ft}.pkl'),

    }

    severities = ["all","v_l", "l", "m", "h", "c"]
    cosines = dict()
    ckas = dict()
    pwccas = dict()
    svccas = dict()

    for sev in severities:
        print(sev)
        transcripts_c = ds["transcripts_c"]
        transcripts_sev = ds[f"transcripts_{sev}"]

        # find corresponding control samples to sev samples
        ids = []
        for i,transcript in enumerate(transcripts_sev):
            if i < len(transcripts_c):
                if transcript == transcripts_c[i]:
                    ids.append(i)
                else:
                    indices = [i for i, x in enumerate(transcripts_c) if x == transcript]
                    if len(indices) == 0:
                        print("error")
                    ind = random.choice(indices)
                    ids.append(ind)
            else:
                indices = [i for i, x in enumerate(transcripts_c) if x == transcript]
                if len(indices) == 0:
                    print("error")
                ind = random.choice(indices)
                ids.append(ind)

        print("cosine")
        cosine = cosine_matrix(infos["c"], infos[sev], ids)
        cosines[sev] = cosine["results"]
        cosines[f"{sev}_b"] = cosine["baseline"]

        print("pwcca")
        pwcca = pwcca_matrix(infos["c"], infos[sev], ids)
        pwccas[sev] = pwcca["results"]
        pwccas[f"{sev}_b"] = pwcca["baseline"]

        print("svcca")
        svcca = svcca_matrix(infos["c"], infos[sev], ids)
        svccas[sev] = svcca["results"]
        svccas[f"{sev}_b"] = svcca["baseline"]

        print("cka")
        cka, cka_baseline = cka_matrix(infos["c"], infos[sev], ids)
        ckas[sev] = cka
        ckas[f"{sev}_b"] = cka_baseline

    with open(f'/sims/cosines_{tavg_favg_att}_c_{pt_ft}.pkl', 'wb') as outp:
        dill.dump(cosines, outp)
    with open(f'/sims/ckas_{tavg_favg_att}_c_{pt_ft}.pkl', 'wb') as outp:
        dill.dump(ckas, outp)
    with open(f'/sims/pwccas_{tavg_favg_att}_c_{pt_ft}.pkl', 'wb') as outp:
        dill.dump(pwccas, outp)
    with open(f'/sims/svccas_{tavg_favg_att}_c_{pt_ft}.pkl', 'wb') as outp:
        dill.dump(svccas, outp)

# dist / sims between pt and ft for all severities separate
def compare_pt_ft(tavg_favg_att, blockid=None):
    if blockid is None:
        blockid = ""
    else:
        blockid = "_"+blockid

    print("block id", blockid)
    infos_pt = {
        'all': load_pkl(f'/{tavg_favg_att}/final_all_pt{blockid}.pkl'),
        'v_l': load_pkl(f'/{tavg_favg_att}/final_v_l_pt{blockid}.pkl'),
        'l': load_pkl(f'/{tavg_favg_att}/final_l_pt{blockid}.pkl'),
        'm': load_pkl(f'/{tavg_favg_att}/final_m_pt{blockid}.pkl'),
        'h': load_pkl(f'/{tavg_favg_att}/final_h_pt{blockid}.pkl'),
        'c': load_pkl(f'/{tavg_favg_att}/final_c_pt{blockid}.pkl'),

    }

    infos_ft = {
        'all': load_pkl(f'/{tavg_favg_att}/final_all_ft{blockid}.pkl'),
        'v_l': load_pkl(f'/{tavg_favg_att}/final_v_l_ft{blockid}.pkl'),
        'l': load_pkl(f'/{tavg_favg_att}/final_l_ft{blockid}.pkl'),
        'm': load_pkl(f'/{tavg_favg_att}/final_m_ft{blockid}.pkl'),
        'h': load_pkl(f'/{tavg_favg_att}/final_h_ft{blockid}.pkl'),
        'c': load_pkl(f'/{tavg_favg_att}/final_c_ft{blockid}.pkl'),
    }

    severities = ["all","v_l", "l", "m", "h", "c"]
    combinations = ["ptft", "ptpt", "ftft"]
    informations = [[infos_pt,infos_ft], [infos_pt, infos_pt], [infos_ft,infos_ft]]
    
    for i,infos in enumerate(informations):
        cosines = dict()
        ckas = dict()
        pwccas = dict()
        svccas = dict()

        for sev in severities:
            print(sev)

            # list of all sample ids
            ids = list(range(infos[0][sev].shape[1]))

            print("cosine")
            cosine = cosine_matrix(infos[0][sev], infos[1][sev], ids)
            cosines[sev] = cosine["results"]
            cosines[f"{sev}_b"] = cosine["baseline"]



            print("pwcca")
            pwcca = pwcca_matrix(infos[0][sev], infos[1][sev], ids)
            pwccas[sev] = pwcca["results"]
            pwccas[f"{sev}_b"] = pwcca["baseline"]

            print("svcca")
            svcca = svcca_matrix(infos[0][sev], infos[1][sev], ids)
            svccas[sev] = svcca["results"]
            svccas[f"{sev}_b"] = svcca["baseline"]

            print("cka")
            cka, cka_baseline = cka_matrix(infos[0][sev], infos[1][sev], ids)
            ckas[sev] = cka
            ckas[f"{sev}_b"] = cka_baseline

        with open(f'/sims/cosines_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(cosines, outp)
        with open(f'/sims/ckas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(ckas, outp)
        with open(f'/sims/pwccas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(pwccas, outp)
        with open(f'/sims/svccas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(svccas, outp)

def compare_pt_ft_subsets(tavg_favg_att, blockid=None):
    if blockid is None:
        blockid = ""
    else:
        blockid = "_"+blockid

    print("block id", blockid)
    infos_pt = {
        'all': load_pkl(f'/{tavg_favg_att}/final_all_pt{blockid}.pkl'),
        'v_l': load_pkl(f'/{tavg_favg_att}/final_v_l_pt{blockid}.pkl'),
        'l': load_pkl(f'/{tavg_favg_att}/final_l_pt{blockid}.pkl'),
        'm': load_pkl(f'/{tavg_favg_att}/final_m_pt{blockid}.pkl'),
        'h': load_pkl(f'/{tavg_favg_att}/final_h_pt{blockid}.pkl'),
        'c': load_pkl(f'/{tavg_favg_att}/final_c_pt{blockid}.pkl'),

    }

    infos_ft = {
        'all': load_pkl(f'/{tavg_favg_att}/final_all_ft{blockid}.pkl'),
        'v_l': load_pkl(f'/{tavg_favg_att}/final_v_l_ft{blockid}.pkl'),
        'l': load_pkl(f'/{tavg_favg_att}/final_l_ft{blockid}.pkl'),
        'm': load_pkl(f'/{tavg_favg_att}/final_m_ft{blockid}.pkl'),
        'h': load_pkl(f'/{tavg_favg_att}/final_h_ft{blockid}.pkl'),
        'c': load_pkl(f'/{tavg_favg_att}/final_c_ft{blockid}.pkl'),
    }

    severities = ["all","v_l", "l", "m", "h", "c"]
    combinations = ["ptft"]
    informations = [[infos_pt, infos_ft]]
    for i,infos in enumerate(informations):
        cosines = dict()
        ckas = dict()
        pwccas = dict()
        svccas = dict()


        for sev in severities:
            # 10 subsets
            for j in range(10):
                print(sev,j)
                randinds = np.random.randint(infos[0][sev].shape[1], size=int(infos[0][sev].shape[1] * 0.85))

                # take random subset
                infos0 = [infos[0][sev][:,k:k+1,:] for k in randinds]
                infos0 = np.concatenate(infos0, axis=1)
                infos1 = [infos[1][sev][:, k:k + 1, :] for k in randinds]
                infos1 = np.concatenate(infos1, axis=1)


                # list of all sample ids
                ids = list(range(infos0.shape[1]))

                print("cosine")
                cosine = cosine_matrix(infos0, infos1, ids)
                cosines[f"{sev}_{j}"] = cosine["results"]
                cosines[f"{sev}_b_{j}"] = cosine["baseline"]

                print("pwcca")
                pwcca = pwcca_matrix(infos0, infos1, ids)
                pwccas[f"{sev}_{j}"] = pwcca["results"]
                pwccas[f"{sev}_b_{j}"] = pwcca["baseline"]

                print("svcca")
                svcca = svcca_matrix(infos0, infos1, ids)
                svccas[f"{sev}_{j}"] = svcca["results"]
                svccas[f"{sev}_b_{j}"] = svcca["baseline"]

                print("cka")
                cka, cka_baseline = cka_matrix(infos0, infos1, ids)
                ckas[f"{sev}_{j}"] = cka
                ckas[f"{sev}_b_{j}"] = cka_baseline

        with open(f'/sims/cosines_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(cosines, outp)
        with open(f'/sims/ckas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(ckas, outp)
        with open(f'/sims/pwccas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(pwccas, outp)
        with open(f'/sims/svccas_{tavg_favg_att}_{combinations[i]}{blockid}.pkl', 'wb') as outp:
            dill.dump(svccas, outp)


if __name__ == "__main__":
    #-----------compare pt&ft, pt&pt, ft&ft tavgs
    compare_pt_ft("tavg")
    #compare_pt_ft_subsets("tavg","B1")
    #compare_pt_ft_subsets("tavg", "B2")
    #compare_pt_ft_subsets("tavg", "B3")
    #compare_pt_ft("tavg", "B2")
    #compare_pt_ft("tavg", "B3")
    compare_pt_ft("favg")
    #-----------comparept&ft, pt&pt, ft&ft atts
    compare_pt_ft("attentions")
    #compare_pt_ft("attentions", "B2")
    #compare_pt_ft("attentions", "B3")
    #-----------compare to control
    compare_to_control("attentions", "pt")
    compare_to_control("attentions", "ft")
    compare_to_control("tavg", "pt")
    compare_to_control("tavg", "ft")
    #compare_to_control("favg", "pt")
    #compare_to_control("favg", "ft")




