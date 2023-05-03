import os
import json
import argparse
import numpy as np
import torch

from utils.text_loader import write_vocabs, load_vocabs_from_pairs
from utils.helper import dict2clsattr
from evals.word_translation import read_txt_embeddings


os.environ['TOKENIZERS_PARALLELISM'] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tgt_lang", default='fr', type=str, required=True)
parser.add_argument("-e", "--emb_type", default='fasttext', type=str)
parser.add_argument("-s", "--src_lang", default='en', type=str, required=True)

word_data = "wiki"
args = parser.parse_args()

with open("configs/settings.json") as f:
    model_config = json.load(f)

args = dict2clsattr(vars(args), model_config)
args.langs = dict(src=args.src_lang, tgt=args.tgt_lang)
args.device = torch.device('cpu')

if args.emb_type == 'fasttext':
    EMB = 'wiki'
    EMB_NAME = 'fasttext'
else:
    EMB = 'htw'
    EMB_NAME = 'htw'


orig_path = f'../dicts/texts/{word_data}/orig_{word_data}_{args.langs["src"]}_{args.langs["tgt"]}_test.txt'
word2ids, embs, id2words = dict(), dict(), dict()

for key, lang in args.langs.items():
    emb_pth = f'../datasets/{EMB}/{EMB}.{lang}.vec'
    id2words[key], word2ids[key], embs[key] = read_txt_embeddings(emb_pth)

root = f"../dicts/embeddings/{EMB_NAME}/{word_data}/{EMB_NAME}_{word_data}_{args.langs['src']}_{args.langs['tgt']}"

htw_embs = dict()
htw_vocabs = dict()
muse_vocabs = load_vocabs_from_pairs(args.langs, args.word_data, args.data_mode, path=orig_path)

for key, lang in args.langs.items():
    inds = list()
    words = list()
    for i, w in enumerate(muse_vocabs[key]):
        if w in word2ids[key]:
            inds.append(word2ids[key][w])
            words.append(w)
    htw_vocabs[key] = words
    htw_embs[key] = embs[key][np.asarray(inds)]

    write_vocabs(words, args.langs[key], args.langs, args.word_data, args.data_mode)
    np.save(root + f"_{args.langs[key]}_test", htw_embs[key])
    print('Done', lang, len(inds))

if EMB == 'htw':
    from utils.helper import generate_path
    fpath = generate_path('txt_pair', dict(word_data=word_data, src_lang=args.langs['src'], tgt_lang=args.langs['tgt'],
                                           data_mode=args.data_mode))
    vocabs = dict(src=list(), tgt=list())

    with open(orig_path) as f:
        lines = f.readlines()

    inds = list()
    new_lines = list()

    for i, l in enumerate(lines):
        x, y = l.strip().lower().split()
        if x in htw_vocabs['src'] and y in htw_vocabs['tgt']:
            inds.append(i)
            new_lines.append(l)

    f = open(fpath, "w")
    for l in new_lines:
        f.write(l)
    f.close()
