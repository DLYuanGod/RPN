import torch.nn as nn
from transformers import RobertaModel,RobertaForMultipleChoice,RobertaForSequenceClassification
from transformers import BertTokenizer, BertModel
import torch

class RobertaClassifier(nn.Module):
    def __init__(self, ):
        super(RobertaClassifier, self).__init__()
        D_in,D_out = 768,20
        self.Roberta = BertModel.from_pretrained('bert-base-uncased')
        self.word_embeddings = self.Roberta.embeddings.word_embeddings
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out), 
        )

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None):
        if inputs_embeds == None:
            outputs = self.Roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        elif attention_mask == None:
            outputs = self.Roberta(input_ids=input_ids)
        else :
            outputs = self.Roberta(attention_mask=attention_mask,
                                inputs_embeds = inputs_embeds)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

class MultipleChoice(nn.Module):
    def __init__(self, ):
        super(MultipleChoice, self).__init__()
        self.Roberta = RobertaForMultipleChoice.from_pretrained('roberta-base')
        self.word_embeddings = self.Roberta.roberta.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None):
        if inputs_embeds == None:
            outputs = self.Roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        elif attention_mask == None:
            outputs = self.Roberta(input_ids=input_ids)
        else :
            outputs = self.Roberta(attention_mask=attention_mask,
                                inputs_embeds = inputs_embeds)
        return outputs

class Classification(nn.Module):
    def __init__(self, numbout):
        super(Classification, self).__init__()
        self.numbout = numbout
        self.Roberta = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels = self.numbout)
        self.word_embeddings = self.Roberta.roberta.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None):
        if inputs_embeds == None:
            outputs = self.Roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        elif attention_mask == None:
            outputs = self.Roberta(input_ids=input_ids)
        else :
            outputs = self.Roberta(attention_mask=attention_mask,
                                inputs_embeds = inputs_embeds)
        return outputs


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self , output_size, embedding_dim=768, filter_num=100, kernel_list=(10, 20, 30), dropout=0.3):
        super(TextCNN, self).__init__()
        self.Roberta = RobertaModel.from_pretrained('roberta-base')
        self.word_embeddings = self.Roberta.embeddings.word_embeddings
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((50 - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input_ids, attention_mask=None, inputs_embeds=None):
        if inputs_embeds == None:
            x = self.word_embeddings(input_ids)
        else:
            x = input_ids
        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits