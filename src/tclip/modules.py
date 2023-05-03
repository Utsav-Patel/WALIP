from torch import nn
from transformers import BertForMaskedLM, BertConfig, AutoConfig, AutoModel
from transformers import ViTModel, ViTConfig

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=256, dropout_prob=0.4):
        super().__init__()
        self.GeLU = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.proj2proj = nn.Linear(projection_dim, projection_dim)
        self.embd2proj = nn.Linear(embedding_dim, projection_dim)
        self.proj_layernorm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x_projected = self.embd2proj(x)
        x = self.GeLU(x_projected)
        x = self.proj2proj(x)
        x = self.dropout_layer(x)
        x = self.proj_layernorm(x + x_projected)
        return x


class ViTImageEncoder(nn.Module):
    def __init__(
        self, model_name='google/vit-base-patch16-224-in21k', pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            config_vit = ViTConfig()
            self.model = ViTModel(config_vit)
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.tt_idx = 0

    def forward(self, x):
        x = self.model(x)
        x = x.last_hidden_state
        return x[:, self.tt_idx, :]


class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, trainable=True):
        super().__init__()
        self.tt_idx = 0
        
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        if pretrained:
            self.model = BertForMaskedLM.from_pretrained(model_name, config=config)
        else:
            self.model = BertForMaskedLM(config)
                
        for param in self.model.parameters():
            param.requires_grad = trainable


    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.hidden_states[-1]
        x = x[:, self.tt_idx, :]
        return x
