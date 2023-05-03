import argparse, json
import numpy as np
import os, torch, sys
import configs

from models.embedding import ClipEmbedding
from translation import align_words, load_test_dict, train_supervision, calculate_similarity, robust_procrustes, get_recall
from evals.word_translation import read_txt_embeddings
from utils.helper import dict2clsattr, generate_path
from utils.text_loader import get_word2id, load_vocabs, load_vocabs_from_pairs, write_vocabs


os.environ['TOKENIZERS_PARALLELISM'] = "false"

parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("-s", "--src_lang", default='en', type=str, required=True)
parser.add_argument("-t", "--tgt_lang", default='ru', type=str, required=True)
parser.add_argument("-w", "--work_mode", type=str, default='c',
                    help="Working mode [b: baseline, c: cuwt, s: supervision")
parser.add_argument("-b", "--baseline", type=str, default='muse', help="Working mode [muse, muve, globe, nn")
parser.add_argument("-p", "--cuwt_phase", type=str, default='t',
                    help="Working phase [c: cluster, u: unsupervised, t: translation")
parser.add_argument("-m", "--translation", type=str, default='s',
                    help="Working phase [s: semi, z: zero-shot, t: transition, z:random")
parser.add_argument("--proc", type=str, default='robust', help='[robust, normal]')
parser.add_argument("-d", "--debug", default=0, type=int)
parser.add_argument("-g", "--gpu_id", default=1, type=int)

working_modes = {'b': 'baseline', 'c': 'cuwt', 's': 'supervision'}
phases = {"character_frequency": "character_frequency", "substring_matching": "substring_matching", 'c': 'cluster',
          'u': 'unsupervised', 't': 'translation'}

######  Load configurations ##############
args = parser.parse_args()

# Load general config
with open(f"configs/settings.json") as f:
    general_configs = json.load(f)
# Load working config
with open(f"configs/{working_modes[args.work_mode]}.json") as f:
    working_configs = json.load(f)
if 'c' == args.work_mode:
    print(working_configs)
    working_configs = working_configs[phases[args.cuwt_phase]]

model_configs = {**general_configs, **working_configs}
args = dict2clsattr(vars(args), model_configs)
# Load langs config
with open(f"configs/langs.json") as f:
    langs_configs = json.load(f)
args.langs = {'src': args.src_lang, 'tgt': args.tgt_lang}
args.lang_configs = {args.src_lang: langs_configs[args.src_lang], args.tgt_lang: langs_configs[args.tgt_lang]}
args.large_model = langs_configs[args.tgt_lang]["large_model"]

if args.gpu_id == -1 or not (torch.cuda.is_available()):
    args.device = torch.device('cpu')
else:
    args.device = torch.device(f'cuda:{args.gpu_id}')


################# ======= Functions ========== ################
def get_mutual_nn(src_embs, tgt_embs, dist_fn, k=5):
    """
    This function takes in two sets of embeddings (src_embs and tgt_embs)
    and calculates mutual nearest neighbors between them using a given
    distance function (dist_fn).

    Parameters:
    src_embs (numpy.ndarray): A numpy array of embeddings for the source set.
    tgt_embs (numpy.ndarray): A numpy array of embeddings for the target set.
    dist_fn (function): A function that takes in two embeddings and returns a distance score.
    k (int): The number of nearest neighbors to consider for each embedding. Default value is 5.

    Returns:
    pair_indices (numpy.ndarray): A numpy array containing the indices of the mutual nearest neighbors.
    filtered_src_embs (numpy.ndarray): A numpy array of the embeddings from the source set that have mutual nearest neighbors.
    filtered_tgt_embs (numpy.ndarray): A numpy array of the embeddings from the target set that have mutual nearest neighbors.
    """
    # Get the indices of the embeddings in each set
    src_idxs = np.arange(len(src_embs))
    tgt_idxs = np.arange(len(tgt_embs))

    # Initialize an empty list to store the mutual nearest neighbors
    mutual_nn_pairs = []

    # For each embedding in the source set, find the k nearest neighbors in the target set
    for src_idx in range(len(src_embs)):
        tgt_nn = sorted(tgt_idxs, key=lambda j: dist_fn(src_embs[src_idx], tgt_embs[j]))[:k]

        # For each nearest neighbor in the target set, find the k nearest neighbors in the source set
        for tgt_idx in tgt_nn:
            src_nn = sorted(src_idxs, key=lambda j: dist_fn(tgt_embs[tgt_idx], src_embs[j]))[:k]

            # If the original source embedding is among the nearest neighbors of the target embedding,
            # then add the pair to the list of mutual nearest neighbors
            for idx in src_nn:
                if idx == src_idx:
                    mutual_nn_pairs.append((src_idx, tgt_idx))

    # Convert the list of mutual nearest neighbors to a numpy array
    pair_idxs = np.array(mutual_nn_pairs)

    # Extract the filtered embeddings from the source and target sets
    filtered_src_embs = src_embs[pair_idxs[:, 0]]
    filtered_tgt_embs = tgt_embs[pair_idxs[:, 1]]

    # Return the indices and filtered embeddings
    return pair_idxs, filtered_src_embs, filtered_tgt_embs


def load_word2ids(languages, word_data, data_mode):
    # Initialize an empty dictionary to store the word-to-ID mappings for each language
    word2ids = {}

    # Iterate through each language in the dictionary of languages
    for key, lang in languages.items():
        # Load the vocabulary for the current language
        vocab = load_vocabs(lang, languages, word_data, data_mode)

        # Get the word-to-ID mapping for the current language and add it to the word2ids dictionary
        word2ids[key] = get_word2id(vocab)

    # Return the word-to-ID mappings for all languages
    return word2ids


def load_embedding(lang, emb_type, word_data, data_mode, opts):
    """
    Load embeddings for a given language and embedding type.

    Parameters:
    lang (str): The language for which embeddings should be loaded.
    emb_type (str): The type of embeddings to load (e.g. 'fasttext', 'glove', etc.).
    word_data (dict): A dictionary containing word data.
    data_mode (str): The mode of the data (e.g. 'train', 'dev', 'test', etc.).
    opts (argparse.Namespace): A namespace containing program options.

    Returns:
    word2ids (dict): A dictionary containing word to ID mappings for the language.
    embs (torch.Tensor): A tensor containing the loaded embeddings.
    """
    vocab = load_vocabs(lang, opts.langs, word_data, data_mode)
    emb_obj = ClipEmbedding(emb_type, lang, data_mode, opts)
    embs = torch.from_numpy(emb_obj.load_embedding(vocab)).to(opts.device)
    word2ids = get_word2id(vocab)
    return word2ids, embs


def load_two_embeddings(languages, emb_type, word_data, data_mode, opts):
    """
    Load embeddings for two languages.

    Parameters:
    languages (dict): A dictionary containing the names of the two languages and their corresponding embeddings.
    emb_type (str): The type of embeddings to load (e.g. 'fasttext', 'glove', etc.).
    word_data (dict): A dictionary containing word data.
    data_mode (str): The mode of the data (e.g. 'train', 'dev', 'test', etc.).
    opts (argparse.Namespace): A namespace containing program options.

    Returns:
    word2ids (dict): A dictionary containing word to ID mappings for both languages.
    embs (dict): A dictionary containing tensors of the loaded embeddings for both languages.
    """
    word2ids, embs = dict(), dict()
    for key, lang in languages.items():
        word2ids[key], embs[key] = load_embedding(lang, emb_type, word_data, data_mode, opts)
    return word2ids, embs


###### Clustering Nouns


def get_dico_dict(dico):
    """
    Converts a tensor of dictionary indices to a Python dictionary.

    Args:
        dico (torch.Tensor): A 2D tensor of shape (n, 2) where each row contains indices of a dictionary entry.

    Returns:
        dict: A Python dictionary where keys are the first column of `dico` and values are the second column of `dico`.
    """
    dictionary = {}
    tensor = dico.cpu().numpy()
    for i in range(len(tensor)):
        row = tensor[i, :]
        if row[0] in dictionary:
            dictionary[row[0]].append(row[1])
        else:
            dictionary[row[0]] = [row[1]]
    return dictionary


def check_correct_pair(indices, dico):
    """
    Checks if the indices in `indices` form a correct dictionary pair based on the `dico` tensor.

    Args:
        indices (dict): A dictionary containing two keys 'src' and 'tgt', each containing a list of indices.
        dico (torch.Tensor): A 2D tensor of shape (n, 2) where each row contains indices of a dictionary entry.

    Returns:
        tuple: A tuple containing:
            - dict: A Python dictionary where keys are the first column of `dico` and values are the second column of `dico`.
            - list: A list of correct dictionary pairs where each entry is a list containing two indices.
    """
    dictionary = get_dico_dict(dico)
    count = 0
    correct_pairs = []
    for i in range(len(indices['src'])):
        x, y = indices['src'][i], indices['tgt'][i]
        if x in dictionary and y in dictionary[x]:
            count += 1
            correct_pairs.append([x, y])
    print(' === Noun Acc: {:.2f}% =='.format(100 * count / len(indices['src'])))
    return dictionary, correct_pairs


def convert_index(test_dico, args):
    """
    Convert the source language indices in `test_dico` to a continuous range of indices
    starting from 0, and return the resulting tensor along with a list of the original
    source language indices and a dictionary mapping them to the new indices.

    Args:
        test_dico (torch.Tensor): A 2D tensor of shape (num_pairs, 2) containing pairs
            of source and target language indices.
        args: A namespace containing various configuration options.

    Returns:
        A tuple (src_test_dico, src_ids_list, src_id_map) where `src_test_dico` is a
        tensor of the same shape as `test_dico` but with the source language indices
        replaced by their new indices, `src_ids_list` is a list of the original source
        language indices, and `src_id_map` is a dictionary mapping the original source
        language indices to their new indices.
    """
    src_ids_list = list()
    for i in test_dico[:, 0].cpu().numpy():
        if i not in src_ids_list:
            src_ids_list.append(i)

    src_id_map = dict()
    for i in range(len(src_ids_list)):
        src_id_map[src_ids_list[i]] = i

    src_test_dico_list = list()
    for i in range(len(test_dico)):
        r = test_dico[i].cpu().numpy()
        src_test_dico_list.append([src_id_map[r[0]], r[1]])
    src_test_dico = torch.Tensor(np.asarray(src_test_dico_list)).type(torch.LongTensor).to(args.device)

    return src_test_dico, src_ids_list, src_id_map


def get_indices_from_nouns(inds, word2ids, src_id_map, args):
    """
    Given a tensor `inds` of shape (num_pairs, 2) containing pairs of source and target
    language indices, return a dictionary of source and target language indices that
    correspond to the nouns in the pairs.

    Args:
        inds (torch.Tensor): A 2D tensor of shape (num_pairs, 2) containing pairs of
            source and target language indices.
        word2ids (dict): A dictionary containing mappings from words to their respective
            indices in the source and target language vocabularies.
        src_id_map (dict): A dictionary mapping the original source language indices to
            their new indices.
        args: A namespace containing various configuration options.

    Returns:
        A dictionary with keys 'src' and 'tgt', each containing a list of indices
        corresponding to the source and target language nouns in the pairs, respectively.
    """
    nouns = dict()
    for lang in ['src', 'tgt']:
        col = 0 if lang == 'src' else 1
        vocab = load_vocabs(args.langs[lang], args.langs, args.word_data, args.data_mode)
        nouns[lang] = [vocab[i] for i in inds[:, col]]

    indices = {'src': list(), 'tgt': list()}
    for i in range(len(nouns['src'])):
        word1, word2 = nouns['src'][i], nouns['tgt'][i]
        if word1 in word2ids['src'] and word2 in word2ids['tgt'] and word2ids['src'][word1] in src_id_map:
            indices['src'].append(src_id_map[word2ids['src'][word1]])
            indices['tgt'].append(word2ids['tgt'][word2])
    return indices


def num_common_chars(str1: str, str2: str) -> set:
    """
    Returns a set containing the characters that are common between two input strings.

    Args:
    - str1 (str): The first input string.
    - str2 (str): The second input string.

    Returns:
    - A set containing the characters that are common between `str1` and `str2`.
    """
    set1 = set(str1)
    set2 = set(str2)
    return set1 & set2


def lcs(str1: str, str2: str) -> str:
    """
    Returns the longest common substring of two input strings.

    Args:
    - str1 (str): The first input string.
    - str2 (str): The second input string.

    Returns:
    - A string representing the longest common substring of `str1` and `str2`.
    """
    m = [[0] * (1 + len(str2)) for i in range(1 + len(str1))]
    longest_len, end_idx = 0, 0
    for i in range(1, 1 + len(str1)):
        for j in range(1, 1 + len(str2)):
            if str1[i - 1] == str2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest_len:
                    longest_len = m[i][j]
                    end_idx = i
            else:
                m[i][j] = 0
    return str1[end_idx - longest_len: end_idx]


###============= Working Mode =============##########
if args.work_mode == 'c':  # CUWT
    ###============= Clustering =============##########
    if args.cuwt_phase == "substring_matching":
        # Set threshold for common substring length
        threshold = 0.75

        # Define noun data and file path to save indices
        noun_data = args.word_data + "_noun"
        lst_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'

        # Load embeddings and vocabularies
        _, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
        src_vocab = load_vocabs(args.langs['src'], args.langs, args.word_data, args.data_mode)
        tgt_vocab = load_vocabs(args.langs['tgt'], args.langs, args.word_data, args.data_mode)

        # Find pairs of words with common substrings and collect data
        pairs = list()
        indices = list()
        src_nouns = list()
        tgt_nouns = list()
        src_indices = list()
        tgt_indices = list()
        for i, src_word in enumerate(src_vocab):
            matches = [(src_word, tgt, i, j) for (j, tgt) in enumerate(tgt_vocab) if
                       len(lcs(src_word, tgt)) >= threshold * max(len(src_word), len(tgt))]
            for match in matches:
                indices.append([match[2], match[3]])
                if match[0] not in src_nouns:
                    src_indices.append(match[2])
                    src_nouns.append(match[0])
                if match[1] not in tgt_nouns:
                    tgt_indices.append(match[3])
                    tgt_nouns.append(match[1])

        # Save indices and vocabularies
        np.save(lst_path, np.array(indices))
        print(f'Found {len(indices)} substring matches')
        write_vocabs(src_nouns, args.langs['src'], args.langs, noun_data, args.data_mode)
        write_vocabs(tgt_nouns, args.langs['tgt'], args.langs, noun_data, args.data_mode)

        # Save embeddings for source and target nouns
        for language, nouns, indices in [('src', src_nouns, src_indices), ('tgt', tgt_nouns, tgt_indices)]:
            emb_path = generate_path('emb_fp', {'lang': args.langs[language], 'src_lang': args.src_lang,
                                                'tgt_lang': args.tgt_lang, 'word_data': noun_data,
                                                'image_data': args.image_data, 'data_mode': 'test',
                                                'selected': args.using_filtered_images, 'num_images': args.num_images})
            np.save(emb_path, embs[language].cpu().numpy())

        # Exit program
        sys.exit("Finished substring matching")

    elif args.cuwt_phase == 'c':  # cluster
        # load embeddings for both source and target languages
        _, embeddings = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)

        # loop through source and target languages
        for language in ['src', 'tgt']:
            # load vocabulary for the current language
            vocabularies = load_vocabs(args.langs[language], args.langs, args.word_data, args.data_mode)

            # get the embedding matrix for the current language
            embedding_matrix = embeddings[language]  # n x m

            # calculate the maximum cosine similarity between embeddings
            max_cosine = torch.topk(embedding_matrix, 2, dim=1)[0].sum(dim=1).cpu().numpy()

            # calculate the median of the maximum cosine similarity
            median = np.quantile(max_cosine, 0.5)

            # get the indices of embeddings that have a maximum cosine similarity greater than the median
            indices = np.where(max_cosine > median)[0]

            # initialize empty lists to store nouns and their corresponding indices
            nouns, new_indices = list(), list()

            # loop through the indices of embeddings
            for index in indices:
                # if the corresponding vocabulary is not already in the list of nouns
                if vocabularies[index] not in nouns:
                    # add the index and vocabulary to their respective lists
                    new_indices.append(index)
                    nouns.append(vocabularies[index])

            # output noun vocabs
            noun_embeddings = embedding_matrix[new_indices]
            noun_data = args.word_data + '_noun'
            write_vocabs(nouns, args.langs[language], args.langs, noun_data, args.data_mode)

            # save embedding matrix
            embedding_path = generate_path('emb_fp',
                                           {'lang': args.langs[language], 'src_lang': args.src_lang,
                                            'tgt_lang': args.tgt_lang,
                                            'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test',
                                            'selected': args.using_filtered_images, 'num_images': args.num_images})
            np.save(embedding_path, noun_embeddings.cpu().numpy())
            print('Saved embedding matrix')

        # exit program when finished
        sys.exit("DONE!!!!")

    elif args.cuwt_phase == 'character_frequency':
        # Load word embeddings for source and target languages
        _, embeddings = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)

        # Define a threshold value for minimum matching character frequency
        matching_threshold = 0.75

        # Append "_noun" to the end of the word data variable to create the noun data variable
        noun_data = args.word_data + "_noun"

        # Set the path for the list of indices
        indices_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'

        # Load source and target vocabularies
        src_vocab_original = load_vocabs(args.langs['src'], args.langs, args.word_data, args.data_mode)
        tgt_vocab_original = load_vocabs(args.langs['tgt'], args.langs, args.word_data, args.data_mode)

        # Create copies of the original vocabularies
        src_vocab = src_vocab_original.copy()
        tgt_vocab = tgt_vocab_original.copy()

        # Count the frequency of characters in the source and target vocabularies
        src_character_counts = dict()
        tgt_character_counts = dict()

        for word in src_vocab_original:
            characters = [char for char in word]
            for character in characters:
                if character not in src_character_counts:
                    src_character_counts[character] = 0
                src_character_counts[character] += 1

        for word in tgt_vocab_original:
            characters = [char for char in word]
            for character in characters:
                if character not in tgt_character_counts:
                    tgt_character_counts[character] = 0
                tgt_character_counts[character] += 1

        # Sort the characters in the source and target vocabularies in descending order by their frequency
        src_characters_sorted = sorted(src_character_counts, key=lambda x: src_character_counts[x], reverse=True)
        tgt_characters_sorted = sorted(tgt_character_counts, key=lambda x: tgt_character_counts[x], reverse=True)

        # Create a mapping between characters in the target vocabulary and characters in the source vocabulary
        character_mapping = {}

        for i in range(min(len(src_characters_sorted), len(tgt_characters_sorted))):
            character_mapping[tgt_characters_sorted[i]] = src_characters_sorted[i]

        # Translate the target vocabulary using the character mapping
        tgt_vocab = []

        for word in tgt_vocab_original:
            translated_word = ""
            for character in word:
                if character in character_mapping:
                    translated_word += character_mapping[character]
                else:
                    translated_word += character
            tgt_vocab.append(translated_word)

        # Find matching character pairs between source and target vocabularies
        indices = list()
        src_noun = list()
        tgt_noun = list()
        src_indices = list()
        tgt_indices = list()
        for i, src_word in enumerate(src_vocab):
            matches = [(src_word, tgt, tgt_vocab_original[j], i, j) for (j, tgt) in enumerate(tgt_vocab) if
                       len(lcs(src_word, tgt)) >= matching_threshold * max(len(src_word), len(tgt))]
            for match in matches:
                print(match)
                indices.append([match[3], match[4]])
                if match[0] not in src_noun:
                    src_indices.append(match[2])
                    src_noun.append(match[0])
                if match[1] not in tgt_noun:
                    tgt_indices.append(match[3])
                    tgt_noun.append(match[1])

        np.save(indices_path, np.array(indices))
        print(f"Found {len(indices)} character match pairs")
        write_vocabs(src_vocab_original, args.langs['src'], args.langs, noun_data, args.data_mode)
        write_vocabs(tgt_vocab_original, args.langs['tgt'], args.langs, noun_data, args.data_mode)
        emb_path = generate_path('emb_fp',
                                 {'lang': args.langs['src'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang,
                                  'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test',
                                  'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embeddings['src'].cpu().numpy())
        emb_path = generate_path('emb_fp',
                                 {'lang': args.langs['tgt'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang,
                                  'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test',
                                  'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embeddings['tgt'].cpu().numpy())
        sys.exit("Finished character freq")

    else:  # semi, zero-shot, transition
        ###============= Unsupervised =============##########
        orig_data = args.word_data
        noun_data = args.word_data + '_noun'
        lst_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'
        if args.cuwt_phase == 'u':  # unsupervised
            args.word_data = noun_data
            word2ids, embs = load_two_embeddings(args.langs, args.emb_type, noun_data, args.data_mode, args)
            # id2words
            scores = calculate_similarity(args.sim_score, None, embs, args.emb_type)
            inds = align_words(None, scores, 0.7, k=1)  # 0.5, k=5
            # testing
            args.word_data = orig_data
            word2ids_wiki = load_word2ids(args.langs, args.word_data, args.data_mode)
            test_dico = load_test_dict(args, word2ids_wiki)
            # Noun-data
            args.word_data = noun_data
            src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
            indices = get_indices_from_nouns(inds, word2ids_wiki, src_id_map, args)

            check_correct_pair(indices, src_test_dico)
            if not (args.debug):
                np.save(lst_path, inds)

elif args.work_mode == 'b' and args.translation == 'nn':
    # Set the embedding type to 'fp'
    embedding_type = 'fp'

    # Load word to ID mappings and embeddings for both source and target languages
    word2ids, embs = load_two_embeddings(args.langs, embedding_type, args.word_data, args.data_mode, args)

    # Load the test dictionary for the given languages
    test_dict = load_test_dict(args, word2ids)

    # Convert the test dictionary to index-based format for the source language
    src_test_dict, src_ids, src_id_map = convert_index(test_dict, args)

    # Get the indices of the most similar words in the target language for each word in the source language
    src_w2i = torch.argmax(embs['src'], dim=1).cpu().numpy()
    tgt_i2w = torch.argmax(embs['tgt'], dim=0).cpu().numpy()
    pairs = []
    for i in range(len(src_w2i)):
        img_index = src_w2i[i]
        word_index = tgt_i2w[img_index]
        pairs.append([i, word_index])

    # Compute the accuracy of the mapping
    dico_dict = get_dico_dict(src_test_dict)
    correct_count = 0
    for i in range(len(pairs)):
        if pairs[i][1] in dico_dict[i]:
            correct_count += 1
    accuracy = correct_count / len(pairs) * 100
    print('Accuracy: ', accuracy)
else:
    if args.work_mode == 's':  # supervised mode
        # define root path for saving/loading files
        root_path = f"../dicts/embeddings/fasttext/wiki/fasttext_wiki_{args.langs['src']}_{args.langs['tgt']}"
        # check if validation set already exists, if not create it
        if not os.path.isfile(root_path + f"_{args.langs['src']}_val.npy"):
            # Load vocabulary pairs from MUSE
            print('Loading MUSE vocabulary pairs...')
            vocabs = load_vocabs_from_pairs(args.langs, args.word_data, 'val', duplicate=True)
            word2ids, full_embs = {}, {}
            # Load embeddings for each language
            for key, lang in args.langs.items():
                emb_path = f'../datasets/wiki/wiki.{lang}.vec'
                print('Loading embeddings for', lang, 'from', emb_path)
                _, word2ids[key], full_embs[key] = read_txt_embeddings(emb_path)
            # Get embeddings for the HTW data
            embs = {}
            for key, lang in args.langs.items():
                inds = [word2ids[key][w] for w in vocabs[key]]
                embs[key] = full_embs[key][np.asarray(inds)]
                np.save(root_path + f"_{args.langs[key]}_val", embs[key])
                print('Done', lang, len(inds))
            # Convert embeddings to PyTorch tensors and move to device
            X = {}
            for key in args.langs:
                X[key] = torch.from_numpy(embs[key]).type(torch.FloatTensor).to(args.device)
        else:
            # load embeddings for training
            word2ids, embs = load_two_embeddings(args.langs, args.emb_type, 'wiki', 'val', args)
            test_dico = load_test_dict(args, word2ids)
            for l in ['src', 'tgt']:
                embs[l] = embs[l].type(torch.FloatTensor).to(args.device)
            X = {'src': embs['src'][test_dico[:, 0]], 'tgt': embs['tgt'][test_dico[:, 1]]}
        # train the model using supervised method
        W = train_supervision(X['src'], X['tgt'])

    else:  # baseline method
        # Define file path for saving/loading the mapping matrix
        w_path = f'../results/mapping/baselines/W_{args.src_lang}_{args.tgt_lang}_{args.method}.pth'
        if os.path.isfile(w_path):
            # Load mapping matrix from PyTorch tensor
            W = torch.load(w_path).to(args.device)
        else:
            # Load mapping matrix from Numpy array and convert to PyTorch tensor
            W = np.load(w_path + '.npy', allow_pickle=True)
            W = torch.from_numpy(W).to(args.device)

    # Load embeddings and test dictionary
    word2ids, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
    test_dico = load_test_dict(args, word2ids)
    src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
    for l in ['src', 'tgt']:
        embs[l] = embs[l].type(torch.FloatTensor).to(args.device)

    # Calculate similarity scores and print recall

    scores = calculate_similarity(args.sim_score, test_dico, {'src': embs['src'] @ W.T, 'tgt': embs['tgt']}, args.emb_type)
    print(get_recall(src_test_dico, scores))