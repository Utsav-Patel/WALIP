import torch, os
import torch.nn.functional as F
import numpy as np
from funcy import chunks
from utils.helper import save_images, generate_path, get_basename
from models.templates import prompts, generate_texts
from models.ops import load_models
from tqdm import tqdm
from utils.image_loader import load_image_data
import configs

def extract_fasttext(language, vocabulary):
    from evals.word_translation import read_txt_embeddings
    embeddings_path = f'../datasets/wiki/wiki.{language}.vec'
    print('Loading ', embeddings_path)
    _, word_to_ids, embeddings = read_txt_embeddings(embeddings_path)
    indices = []
    for i, word in enumerate(vocabulary):
        if word in word_to_ids:
            indices.append(word_to_ids[word])

    sub_embeddings = embeddings[np.asarray(indices)]
    print('Extract fasttext', language, len(indices))
    return sub_embeddings

class ClipEmbedding():
    def __init__(self, emb_type, lang, data_mode, opts):
        self.emb_type, self.lang, self.data_mode, self.opts = emb_type, lang, data_mode, opts
        self.lang_opts = opts.lang_configs[lang]
        print(self.lang, emb_type, 'embedding')

        self.emb_path = generate_path('emb_' + emb_type, {'lang': lang, 'src_lang': opts.src_lang, 'tgt_lang': opts.tgt_lang, 'word_data': opts.word_data, 'image_data': opts.image_data, 'data_mode': data_mode, 'selected': opts.using_filtered_images, 'num_images': opts.num_images})
        print(f'Loading embeddings from: {self.emb_path}')
        self.model = None
    
    def load_clip_model(self):
        model, logit_scale, preprocess = load_models(self.lang, device=self.opts.device, large_model=self.lang_opts["large_model"])
        self.model = model
        self.logit_scale = logit_scale
        self.preprocess = preprocess
    
    def set_logit_scale(self, value):
        self.logit_scale = value

    def load_embedding(self, vocabs=None):
        # check embedding reuse
        if self.opts.reuse_emb or self.lang_opts["reuse_emb"]:
            if os.path.isfile(self.emb_path):
                print('.....', 'Reuse emb', get_basename(self.emb_path))
                return np.load(self.emb_path, allow_pickle=True)
            else:
                print("No embedding exists!!!")

        if self.emb_type in [configs.FASTTEXT, configs.GLOBE, configs.HTW]:
            if os.path.isfile(self.emb_path):
                embeddings = np.load(self.emb_path, allow_pickle=True)
            else:
                extract_fasttext(self.lang, vocabs)
        else:
            print('.....', "New embedding", get_basename(self.emb_path))
            txt_embeddings = self.load_clip_txt_emb(vocabs)
            if self.emb_type == configs.FINGERPRINT:
                img_embeddings = self.load_clip_img_emb()
                embeddings = self.load_fingerprint(img_embeddings, torch.from_numpy(txt_embeddings).to(self.opts.device))
            else:
                embeddings = txt_embeddings
            np.save(self.emb_path, embeddings) 
        return embeddings

    def load_clip_img_emb(self):
        img_emb_path = generate_path('emb_img', {'lang': self.lang, 'image_data': self.opts.image_data, 'selected': self.opts.using_filtered_images, 'num_images': self.opts.num_images})
        if self.lang_opts["reuse_img_emb"] and os.path.isfile(img_emb_path):
            print('.....', '.....', "Reuse img emb", get_basename(img_emb_path))
            img_embeddings = np.load(img_emb_path, allow_pickle=True)
            img_embeddings = torch.Tensor(img_embeddings).to(self.opts.device)
        else:
            if self.model is None:
                self.load_clip_model()
            print('.....', '.....', "New image embedding") 

            img_path = generate_path('img', {'lang': self.lang, 'image_data': self.opts.image_data, 'selected': self.opts.using_filtered_images, 'num_images': self.opts.num_images})
            if self.opts.reuse_image_data and os.path.isfile(img_path): 
                print('.....', '.....', '.....', "Reuse img data") 
                images = np.load(img_path, allow_pickle=True)
            else:
                print('.....', '.....', '.....',

