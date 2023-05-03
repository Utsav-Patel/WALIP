
import os
import sys
import configs

from typing import Dict, List, Optional
from utils.helper import generate_path


def load_vocabs(lang: str, langs: Dict[str, str], word_data: str, data_mode: str) -> List[str]:
    """
    Load vocabulary of a single language.

    Args:
        lang (str): Language code.
        langs (Dict[str, str]): A dictionary of source and target language codes.
        word_data (str): Name of the data.
        data_mode (str): Train, valid or test.

    Returns:
        List[str]: A list of vocabulary words.
    """
    file_path = generate_path('txt_single', {'word_data': word_data, 'lang': lang, 'src_lang': langs['src'],
                                             'tgt_lang': langs['tgt'], 'data_mode': data_mode})
    if not os.path.isfile(file_path):
        print(f"------> Error: Load vocabs {file_path} file doesn't exist!!!")
        sys.exit('Done')
    with open(file_path) as file:
        lines = file.readlines()
    vocab = list()
    for line in lines:
        word = line.strip().lower()
        if word not in vocab:
            vocab.append(word)
    print(f'Loaded {len(vocab)} words from {file_path}')
    return vocab


def load_vocabs_from_pairs(langs: Dict[str, str], word_data: str, data_mode: str,
                           duplicate: bool = False, path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load source and target vocabularies from parallel data files.

    Args:
        langs (Dict[str, str]): A dictionary of source and target language codes.
        word_data (str): Name of the parallel data.
        data_mode (str): Train, valid or test.
        duplicate (bool, optional): Whether to include duplicate words in the vocabularies. Defaults to False.
        path (str, optional): Path to the parallel data file. Defaults to None.

    Returns:
        Dict[str, List[str]]: A dictionary containing two lists of source and target vocabularies.
    """
    file_path = generate_path('txt_pair', {'word_data': word_data, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'],
                                           'data_mode': data_mode})
    if path is not None:
        file_path = path
    src_vocab = list()
    tgt_vocab = list()
    with open(file_path) as file:
        lines = file.readlines()
    for line in lines:
        src_word, tgt_word = line.strip().lower().split(configs.delimiters[word_data])
        if duplicate:
            src_vocab.append(src_word)
            tgt_vocab.append(tgt_word)
        else:
            if src_word not in src_vocab:
                src_vocab.append(src_word)
            if tgt_word not in tgt_vocab:
                tgt_vocab.append(tgt_word)
    return {'src': src_vocab, 'tgt': tgt_vocab}


def write_vocabs(vocabs: List[str], lang_code: str, lang_dict: Dict[str, str], word_data_name: str, data_mode_name: str) -> None:
    """
    Write a list of vocabulary words to a file.

    Args:
        vocabs (List[str]): A list of vocabulary words.
        lang_code (str): Language code.
        lang_dict (Dict[str, str]): A dictionary of source and target language codes.
        word_data_name (str): Name of the data.
        data_mode_name (str): Train, valid or test.

    Returns:
        None
    """
    filepath = generate_path('txt_single', {'word_data': word_data_name, 'lang': lang_code, 'src_lang': lang_dict['src'],
                                             'tgt_lang': lang_dict['tgt'], 'data_mode': data_mode_name})
    with open(filepath, "w") as file:
        for word in vocabs:
            file.write(f"{word}\n")


def combine_files(lang_dict: Dict[str, str], word_data_name: str, data_mode_name: str) -> None:
    """
    Combine the vocabulary files of two languages into a parallel vocabulary file.

    Args:
        lang_dict (Dict[str, str]): A dictionary of source and target language codes.
        word_data_name (str): Name of the data.
        data_mode_name (str): Train, valid or test.

    Returns:
        None
    """
    src_vocabs = load_vocabs(lang_dict['src'], lang_dict, word_data_name, data_mode_name)
    tgt_vocabs = load_vocabs(lang_dict['tgt'], lang_dict, word_data_name, data_mode_name)
    filepath = generate_path('txt_pair', {'word_data': word_data_name, 'src_lang': lang_dict['src'], 'tgt_lang': lang_dict['tgt'],
                                           'data_mode': data_mode_name})
    with open(filepath, "w") as file:
        for i in range(len(src_vocabs)):
            file.write(f"{src_vocabs[i]}{configs.delimiters[word_data_name]}{tgt_vocabs[i]}\n")


def get_word2id(vocab: List[str]) -> Dict[str, int]:
    """
    Create a dictionary mapping vocabulary words to their corresponding indices.

    Args:
        vocab (List[str]): A list of vocabulary words.

    Returns:
        Dict[str, int]: A dictionary mapping vocabulary words to their corresponding indices.
    """
    word_to_id = dict()
    for idx, word in enumerate(vocab):
        if word not in word_to_id:
            word_to_id[word] = idx
    return word_to_id
