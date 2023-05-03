import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, AutoTokenizer
from tclip.modules import ViTImageEncoder, TextEncoder, ProjectionHead


def cross_entropy(preds, targets, reduction=None):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "mean":
        loss = loss.mean()
    return loss

def get_tokenizer(language, tokenizer_name):
    tokenizer = BertTokenizer if language == "en" else AutoTokenizer
    tokenizer = tokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
    return tokenizer

class CLIPModel(nn.Module):
    def __init__(
        self, language, tokenizer_name, pretrained=True, temperature=0.07,
        text_embedding=768, device='cuda'
    ):
        super().__init__()
        self.image_encoder = ViTImageEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(tokenizer_name, pretrained=pretrained)

        self.image_projection = ProjectionHead(embedding_dim=768)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

        self.temperature = temperature
        self.max_length = 200

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = device
        self.tokenizer = get_tokenizer(language, tokenizer_name)
        
        # self.encode_image = lambda images: self.image_projection(self.image_encoder(images))

        # self.extr_tokens = lambda texts: self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True)  
        
        # self.encode_text = lambda texts: (
        #     tokens_text := self.extr_tokens(texts),
        #     input_ids := tokens_text["input_ids"].to(device),
        #     attention_mask := tokens_text["attention_mask"].to(device),
        #     encoded_txts := self.text_encoder(input_ids=input_ids, attention_mask=attention_mask),
        #     self.text_projection(encoded_txts)
        # )
        
    def encode_image(self, images):
        encoded_imgs = self.image_encoder(images)
        embed_imgs = self.image_projection(encoded_imgs)
        return embed_imgs

    def encode_text(self, texts):
        tokens_text = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True)
        input_ids = tokens_text["input_ids"].to(self.device)
        attention_mask = tokens_text["attention_mask"].to(self.device)
        encoded_txts = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embed_txts = self.text_projection(encoded_txts)
        return embed_txts


    def get_embeddings(self, batch):
        images, input_ids, attention_mask = batch["images"], batch["attention_mask"], batch["input_ids"]
        encoded_imgs = self.image_encoder(images)
        encoded_txts = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embed_imgs = self.image_projection(encoded_imgs)
        embed_txts = self.text_projection(encoded_txts)
        return embed_imgs, embed_txts

    def forward(self, batch):
        embed_imgs, embed_txts = self.get_embeddings(batch)

        logits = torch.matmul(embed_imgs, embed_txts.T) * np.exp(self.temperature)
        images_similarity = torch.matmul(embed_imgs, embed_imgs.T)
        texts_similarity = torch.matmul(embed_txts, embed_txts.T)

        logits_sim = (images_similarity + texts_similarity) / 2 * np.exp(self.temperature)
        targets = F.softmax(logits_sim, dim=-1)

        texts_loss = cross_entropy(logits.T, targets.T, reduction='mean')
        images_loss = cross_entropy(logits, targets, reduction='mean')
        loss = (images_loss + texts_loss) / 2

        return loss
