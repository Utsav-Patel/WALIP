import numpy as np
import torch
import os
import scipy
import scipy.optimize
import torch.nn.functional as F
from evals.word_translation import get_csls_word_translation, build_dictionary, get_topk_translation_accuracy, load_dictionary
from utils.helper import generate_path
from utils.text_loader import combine_files
import configs


def robust_procrustes(X, Y):
    n, k = X.shape
    W = train_supervision(X, Y)
    D = torch.eye(n).to(X.device)
    X1 = X
    Y1 = Y
    for _ in range(2):
        e = ((Y1 - X1 @ W.T)**2).sum(dim=1)
        alphas = 1 / (e + 0.001)
        alphas = alphas / alphas.max()
        for i in range(n):
            D[i, i] = alphas[i]**0.5
        X1 = D @ X1
        Y1 = D @ Y1
        M = Y1.T @ X1
        U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
        W = U.dot(VT)
    return W


def load_test_dict(params, word2ids):
    test_file_path = generate_path('txt_pair', {'word_data': params.word_data, 'src_lang': params.src_lang, 'tgt_lang': params.tgt_lang, 'data_mode': params.data_mode})
    if not os.path.isfile(test_file_path):
        combine_files(params.langs, params.word_data, params.data_mode)
    test_dict = load_dictionary(test_file_path, word2ids['src'], word2ids['tgt'])
    test_dict = test_dict.to(params.device)
    return test_dict


def align_words(dictionary, scores, c=0.5, k=1):
    if dictionary is not None:
        results = get_topk_translation_accuracy(dictionary, scores)
        print(results)
        dict_dict = get_dictionary_dict(dictionary)

    topk_scores = scores.topk(k, 1, True)

    def get_precision(topk_scores, threshold):
        correct, total, lst = 0, 0, []
        for i in range(len(scores)):
            for m in range(k):
                if topk_scores[0][i, m] > threshold:
                    total += 1
                    lst.append([i, topk_scores[1][i, m].cpu().numpy()])
                    if dictionary is not None and topk_scores[1][i, m] in dict_dict[dictionary[i, 0].item()]:
                        correct += 1
        if dictionary is not None:
            print("---------> Prec@1 {:.2f} {}/{}".format(correct/total*100, correct, total))
        return lst

    if c > 0:
        threshold = torch.quantile(topk_scores[0], c)
    else:
        threshold = 0
    lst = get_precision(topk_scores, threshold)
    return np.asarray(lst)



def train_supervision(X1, X2):
    M = X2.transpose(0, 1).mm(X1).cpu().numpy()
    U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
    W = U.dot(VT)
    return torch.Tensor(W).to(X1.device)


def get_recall(dictionary, scores):
    dict_dict = get_dictionary_dict(dictionary)
    topk_scores = scores.topk(10, 1, True)
    correct1, correct10 = 0, 0
    checked_ids = []
    for i in range(len(scores)):
        k = dictionary[i, 0].item()
        if k not in checked_ids:
            checked_ids.append(k)
            if topk_scores[1][i, 0] in dict_dict[k]:
                correct1 += 1
                correct10 += 1
            else:
                for t in dict_dict[k]:
                    if t in topk_scores[1][i, :]:
                        correct10 += 1
                        break
    total = len(checked_ids)
    return correct1/total*100, correct10/total*100


def get_dictionary_dict(dictionary):
    dict_dict = {}
    tensor_dict = dictionary.cpu().numpy()
    for i in range(len(tensor_dict)):
        row = tensor_dict[i, :]
        if row[0] in dict_dict:
            dict_dict[row[0]].append(row[1])
        else:
            dict_dict[row[0]] = [row[1]]
    return dict_dict


def calculate_similarity(similarity_score, dictionary, embeddings, embedding_type, row_ranking=True):
    for lang in ['source', 'target']:
        if embedding_type == 'fp':
            t = np.quantile(embeddings[lang].cpu().numpy(), 0.9)
            embeddings[lang] = embeddings[lang] * (embeddings[lang] > t)
        embeddings[lang] = F.normalize(embeddings[lang], dim=1)
    scores = get_csls_word_translation(dictionary, embeddings['source'], embeddings['target'], similarity_score)
    return scores