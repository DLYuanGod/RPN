import torch.nn as nn
from transformers import XLNetModel

class XLnetClassifier(nn.Module):
    def __init__(self, ):
        super(XLnetClassifier, self).__init__()
        D_in,D_out = 768,3
        self.XLnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.word_embeddings = self.XLnet.embeddings.word_embeddings
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out),
        )

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        if inputs_embeds == None:
            outputs = self.XLnet(input_ids=input_ids,
                                attention_mask=attention_mask)
        else :
            outputs = self.XLnet(attention_mask=attention_mask,
                                inputs_embeds = inputs_embeds)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits