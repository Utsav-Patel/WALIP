import io
import os, sys
import numpy as np
import torch
from logging import getLogger
import faiss

logger = getLogger()

def nn_average_distance(embeddings, query_vector, k):
    # Convert embeddings and query vector to numpy arrays
    embeddings = embeddings.cpu().numpy()
    query_vector = query_vector.cpu().numpy()

    # Convert embeddings and query vector to float32
    embeddings = np.float32(embeddings)
    query_vector = np.float32(query_vector)

    # Initialize resources and index for GPU processing
    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    index = faiss.GpuIndexFlatIP(res, embeddings.shape[1], config)

    # Add embeddings to the index
    index.add(embeddings)

    # Search for the k nearest neighbors and compute the mean distances
    distances, _ = index.search(query_vector, k)
    return distances.mean(1)


# Build a dictionary of identical character strings.
def load_identical_char_dico(word_dict1, word_dict2):
    """
    Build a dictionary of identical character strings.
    """
    # Create a list of word pairs where the source word exists in both word_dict1 and word_dict2
    pairs = [(w1, w1) for w1 in word_dict1.keys() if w1 in word_dict2]

    # Sort the pairs list by the frequency of the source word in word_dict1
    pairs = sorted(pairs, key=lambda x: word_dict1[x[0]])

    # Create a PyTorch LongTensor to store the word index pairs
    index_pairs = torch.LongTensor(len(pairs), 2)

    # Fill in the index_pairs LongTensor with the word index pairs
    for i, (word1, word2) in enumerate(pairs):
        index_pairs[i, 0] = word_dict1[word1]
        index_pairs[i, 1] = word_dict2[word2]

    return index_pairs


# Return a torch tensor of size (n, 2) where n is the size of the
# loader dictionary, and sort it by source word frequency.
def load_dictionary(path, src_word2id, tgt_word2id, delimiter=None):
    word_pairs = []
    num_not_found = 0
    num_not_found_src = 0
    num_not_found_tgt = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.lower()
            if delimiter is not None:
                parts = line.rstrip().split(delimiter)
            else:
                parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            src_word, tgt_word = parts
            if src_word in src_word2id and tgt_word in tgt_word2id:
                word_pairs.append((src_word, tgt_word))
            else:
                num_not_found += 1
                num_not_found_src += int(src_word not in src_word2id)
                num_not_found_tgt += int(tgt_word not in tgt_word2id)

    # sort the word_pairs list by the frequency of the source word in src_word2id
    word_pairs = sorted(word_pairs, key=lambda x: src_word2id[x[0]])

    # create a PyTorch LongTensor to store the word index pairs
    index_pairs = torch.LongTensor(len(word_pairs), 2)

    # fill in the index_pairs LongTensor with the word index pairs
    for i, (src_word, tgt_word) in enumerate(word_pairs):
        index_pairs[i, 0] = src_word2id[src_word]
        index_pairs[i, 1] = tgt_word2id[tgt_word]

    return index_pairs


# Given source and target word embeddings, and a dictionary,
# evaluate the translation accuracy using the precision@k.
def get_word_translation(index_pairs, emb1, emb2, sim_score='csls'):
    if index_pairs is None:
        query = emb1
    else:
        query = emb1[index_pairs[:, 0]]

    if sim_score == 'cosine':
        scores = query.mm(emb2.transpose(0, 1))

    elif sim_score.startswith('csls'):
        knn = 10
        average_dist1 = nn_avergae_distance(emb2, emb1, knn)
        average_dist2 = nn_avergae_distance(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        if index_pairs is None:
            scores.sub_(average_dist1[:, None])
        else:
            scores.sub_(average_dist1[index_pairs[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    return scores


def get_topk_translation_accuracy(dictionary, similarity_scores):
    top_k_results = []
    top_matches = similarity_scores.topk(10, 1, True)[1]
    for k_val in [1, 5, 10]:
        top_k_matches = top_matches[:, :k_val]
        num_matches = (top_k_matches == dictionary[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        match_dict = {}
        for i, source_id in enumerate(dictionary[:, 0].cpu().numpy()):
            match_dict[source_id] = min(match_dict.get(source_id, 0) + num_matches[i], 1)
        precision_at_k = 100 * np.mean(list(match_dict.values()))
        top_k_results.append('prec_@{}: {:.2f}'.format(k_val, precision_at_k))
    return top_k_results

#Reload pretrained embeddings from a text file.
def read_txt_embeddings(embedding_path, embedding_dim=300, use_full_vocab=False):
    word2id_dict = {}
    embeddings_list = []
    max_vocab_size = 200000
    _embedding_dim = embedding_dim
    with io.open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split_line = line.split()
            else:
                word, vector = line.rstrip().split(' ', 1)
                if not use_full_vocab:
                    word = word.lower()
                vector = np.fromstring(vector, sep=' ')
                if np.linalg.norm(vector) == 0:  # Avoid having null embeddings
                    vector[0] = 0.01
                if word in word2id_dict:
                    if use_full_vocab:
                        print("Word '%s' found twice in embedding file" % (word))
                else:
                    if not vector.shape == (_embedding_dim,):
                        print("Invalid dimension (%i) word '%s' in line %i." % (vector.shape[0], word, i))
                        continue
                    word2id_dict[word] = len(word2id_dict)
                    embeddings_list.append(vector[None])
            if max_vocab_size > 0 and len(word2id_dict) >= max_vocab_size and not use_full_vocab:
                break

    id2word_dict = {v: k for k, v in word2id_dict.items()}
    embeddings_array = np.concatenate(embeddings_list, 0)

    return id2word_dict, word2id_dict, embeddings_array

#Get best translation pairs candidates.
def get_candidates(source_emb, target_emb, params):

    batch_size = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    source_size = source_emb.size(0)
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        source_size = min(params.dico_max_rank, source_size)

    # nearest neighbors
    if params.dico_method == 'nn':

        # for every source word
        for i in range(0, source_size, batch_size):

            # compute target words scores
            scores = target_emb.mm(source_emb[i:min(source_size, i + batch_size)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif params.dico_method.startswith('invsm_beta_'):

        beta = float(params.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, target_emb.size(0), batch_size):

            # compute source words scores
            scores = source_emb.mm(target_emb[i:i + batch_size].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params.dico_method.startswith('csls_knn_'):

        knn = params.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(nn_avergae_distance(target_emb, source_emb, knn))
        average_dist2 = torch.from_numpy(nn_avergae_distance(source_emb, target_emb, knn))
        average_dist1 = average_dist1.type_as(source_emb)
        average_dist2 = average_dist2.type_as(target_emb)

        # for every source word
        for i in range(0, source_size, batch_size):

            # compute target words scores
            scores = target_emb.mm(source_emb[i:min(source_size, i + batch_size)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(source_size, i + batch_size)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores,


# Build a training dictionary given current embeddings / mapping.
def build_dictionary(source_embedding, target_embedding, parameters, source_to_target_candidates=None, target_to_source_candidates=None):
    print("Building the train dictionary ...")
    source_to_target = 'S2T' in parameters.dico_build
    target_to_source = 'T2S' in parameters.dico_build
    assert source_to_target or target_to_source

    if source_to_target:
        if source_to_target_candidates is None:
            source_to_target_candidates = get_candidates(source_embedding, target_embedding, parameters)
    if target_to_source:
        if target_to_source_candidates is None:
            target_to_source_candidates = get_candidates(target_embedding, source_embedding, parameters)
        target_to_source_candidates = torch.cat([target_to_source_candidates[:, 1:], target_to_source_candidates[:, :1]], 1)

    if parameters.dico_build == 'S2T':
        dictionary = source_to_target_candidates
    elif parameters.dico_build == 'T2S':
        dictionary = target_to_source_candidates
    else:
        source_to_target_candidates = set([(a, b) for a, b in source_to_target_candidates.numpy()])
        target_to_source_candidates = set([(a, b) for a, b in target_to_source_candidates.numpy()])
        if parameters.dico_build == 'S2T|T2S':
            final_pairs = source_to_target_candidates | target_to_source_candidates
        else:
            assert parameters.dico_build == 'S2T&T2S'
            final_pairs = source_to_target_candidates & target_to_source_candidates
            if len(final_pairs) == 0:
                print("Empty intersection ...")
                return None
        dictionary = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    print('New train dictionary of {} pairs.'.format(dictionary.size(0)))
    return dictionary.cuda()

