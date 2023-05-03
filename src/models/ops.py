import torch
from tclip.CLIP import CLIPModel as TClip
import numpy as np
import clip
import ruclip


class EnglishClipObject():
    def __init__(self, name="ViT-B/32", device="cuda:0") -> None:
        self.clip_model, self.preprocess = clip.load(name)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.device = device
    
    def encode_image(self, imgs):
        return self.clip_model.encode_image(imgs).type(torch.FloatTensor).to(self.device)

    def encode_text(self, txts):
        text_tokens = clip.tokenize(txts).to(self.device)
        embed_txts = self.clip_model.encode_text(text_tokens).type(torch.FloatTensor).to(self.device)
        return embed_txts


class RuClipObject():
    def __init__(self, name='ruclip-vit-base-patch32-384', device="cuda:0"):
        self.clip_model, self.clip_processor = ruclip.load(name, device=device)
        self.clip_model = self.clip_model.to(device).eval()
        self.logit_scale = self.clip_model.logit_scale.exp().float()
        self.preprocess = self.clip_processor.image_transform
        self.device = device
    
    def encode_image(self, imgs):
        latents = self.clip_model.encode_image(imgs.to(self.device))
        embed_imgs = latents / latents.norm(dim=-1, keepdim=True)
        return embed_imgs

    def encode_text(self, txts):
        inputs = self.clip_processor(text=txts, return_tensors='pt', padding=True)
        embed_txts = self.clip_model.encode_text(inputs['input_ids'].to(self.device))

        return embed_txts


def load_models(language, device='cuda:0', large_model=False, model_dir='../results/clips'):
    model, logit_scale, preprocess = None, None, None
    if not large_model:
        model = TClip(language, None, pretrained=False, device=device)
        checkpoint_path = "%s/best_%s.pt" % (model_dir, language)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.eval()
        logit_scale = np.exp(model.temperature)
    else:
        if language == 'en' or '2' in language:
            model = EnglishClipObject(device=device)
        elif language == 'ru':
            model = RuClipObject(name='ruclip-vit-base-patch32-224', device=device)
        logit_scale = model.logit_scale
        preprocess = model.preprocess
        
    return model, logit_scale, preprocess
