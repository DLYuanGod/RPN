import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler ,RandomSampler
import XLNet_model
import Roberta_models
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import numpy as np
import add_noise as an
import pandas as pd
import building_the_dataset as bd

def building_dataloader(dataset,batch_size,gpus,mode='else'):
    if mode == 'train':
        if gpus>1:
            batch_size = batch_size // gpus
            dist.init_process_group(backend="nccl")
            sampler = DistributedSampler(dataset)
        else:
            sampler=RandomSampler(dataset)
    elif mode == 'else':
        sampler=SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, drop_last=True)
    return dataloader

def loss_function(logits, labels):
    loss_fn = nn.CrossEntropyLoss()
    return (loss_fn(logits,labels))

def initialize_model(models,gpus,local_rank,lr,eps,dataloader_train,epochs,tasks_kinds,num_labels):
    torch.cuda.set_device(local_rank)
    if  models == 'XLNet':
        model = XLNet_model.XLnetClassifier()
        model.cuda()
    if  models == 'Roberta':
        if tasks_kinds == 'Classification':
            model = Roberta_models.Classification(num_labels=num_labels)
        elif tasks_kinds == 'MultipleChoice':
            model = Roberta_models.MultipleChoice()
        else:
            model = Roberta_models.TextCNN(output_size=20)
        model.cuda()
    if gpus>1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda() 
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = AdamW(model.parameters(),lr=lr,eps=eps)
    total_steps = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=0,  # Default value
                                        num_training_steps=total_steps)
    return model, optimizer, scheduler


def evaluate(model, test_dataloader):
    model.eval()

    test_accuracy = []
    test_loss = []

    for batch in test_dataloader:
        input_ids_test, attention_masks_test, labels_test = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            logits = model(input_ids_test, attention_masks_test)
        loss = loss_function(logits, labels_test.long())
        test_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == labels_test).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)


    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)
    print(
            f"{'-':^40} | {val_loss:^12.6f} | {val_accuracy:^12.2f}%")
    print("\n")

def predict(model, test, mask):
    model.eval()
    with torch.no_grad():
            logits = model(test.cuda(),mask.cuda())
    logits = torch.argmax(logits, dim=1).flatten()
    dataframe = pd.DataFrame({'Label':bd.label_encoder(logits.cpu().numpy(),model='decoder') })
    dataframe.index = np.arange(1, len(dataframe)+1)
    dataframe.to_csv("output.csv",sep=',') 

def train(model, train_dataloader, optimizer, scheduler, noise, scale, prob, dataloader_validation = None, adv_lr = 1e-4, adv = True, epochs=5, evaluation=False,gpus=1,adv_step=1):
    for epoch_i in range(epochs):
        #               Training
        if gpus > 1:
            train_dataloader.sampler.set_epoch(epochs)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            input_ids_train, attention_masks_train, labels_train = tuple(t.cuda() for t in batch)
            model.zero_grad()
            logits = model(input_ids_train, attention_masks_train)

            loss = loss_function(logits, labels_train)
            loss = loss.mean()
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            if noise == 'None':
                embeddings = model.module.word_embeddings(input_ids_train)
                delta = torch.zeros_like(embeddings).uniform_(-1, 1) * attention_masks_train.unsqueeze(2)
                input_lengths = torch.sum(attention_masks_train, 1)
                dims = input_lengths * embeddings.size(-1)
                mag = 1 / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
                for adv_s in range(adv_step):
                    delta.requires_grad=True
                    embeddings = embeddings + delta
                    outputs = model(input_ids=input_ids_train,attention_mask=attention_masks_train, inputs_embeds=embeddings)
                    noise_loss = loss_function(outputs, labels_train)
                    noise_loss = noise_loss.mean()
                    noise_loss.backward(retain_graph=True)
                    delta.retain_grad()
                    if adv_s == adv_step - 1:
                            break
                    # delta's graduate
                    delta_grad = delta.grad.clone().detach()
                    # norm
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + adv_lr * delta_grad / denorm).detach()
                    # renew delta
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > 1).to(embeddings)
                    reweights = (1 / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()

            if noise == 'G' or noise == 'SAP' or noise == 'RPN':
                for adv_s in range(adv_step):
                    embeddings = model.module.word_embeddings(input_ids_train)
                    if step == 0:
                        an.selection_mat(embeddings,prob)
                    #Add Gaussian noise
                    if noise == 'G':
                        embeddings = an.Gaussian_noise(embeddings,attention_masks_train,scale)
                    #Add salt and pepper noise
                    elif noise == 'SAP':
                        embeddings = an.SAP_noise(embeddings,prob)
                    #Add Random Position Noise
                    elif noise == 'RPN':
                        embeddings = an.RPN_noise(embeddings)                 
                    outputs = model(input_ids=input_ids_train,attention_mask=attention_masks_train, inputs_embeds=embeddings)
                    noise_loss = loss_function(outputs, labels_train)
                    noise_loss = noise_loss.mean()
                    embeddings.retain_grad()
                    noise_loss.backward(retain_graph=True)

            if noise =='whole':
                embeddings = model.module.word_embeddings(input_ids_train)
                if step == 0:
                        an.selection_mat(embeddings,prob)
                RPN_embeddings = an.RPN_noise(embeddings)
                delta = torch.zeros_like(embeddings).uniform_(-1, 1) * attention_masks_train.unsqueeze(2)
                input_lengths = torch.sum(attention_masks_train, 1)
                dims = input_lengths * embeddings.size(-1)
                mag = 1 / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
                for adv_s in range(adv_step):
                    delta.requires_grad=True
                    embeddings = embeddings + delta
                    outputs = model(input_ids=input_ids_train,attention_mask=attention_masks_train, inputs_embeds=embeddings)
                    noise_loss = loss_function(outputs, labels_train)
                    noise_loss = noise_loss.mean()
                    noise_loss.backward(retain_graph=True)
                    delta.retain_grad()
                    if adv_s == adv_step - 1:
                            break
                    # delta's graduate
                    delta_grad = delta.grad.clone().detach()
                    # norm
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + adv_lr * delta_grad / denorm).detach()
                    # renew delta
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > 1).to(embeddings)
                    reweights = (1 / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()

                delta_RPN = torch.zeros_like(RPN_embeddings).uniform_(-1, 1) * attention_masks_train.unsqueeze(2)
                input_lengths = torch.sum(attention_masks_train, 1)
                dims = input_lengths * RPN_embeddings.size(-1)
                mag = 1 / torch.sqrt(dims)
                delta_RPN = (delta_RPN * mag.view(-1, 1, 1)).detach()
                for adv_s in range(adv_step):
                    delta_RPN.requires_grad=True
                    RPN_embeddings = RPN_embeddings + delta_RPN
                    outputs = model(input_ids=input_ids_train,attention_mask=attention_masks_train, inputs_embeds=RPN_embeddings)
                    noise_loss = loss_function(outputs, labels_train)
                    noise_loss = noise_loss.mean()
                    noise_loss.backward(retain_graph=True)
                    delta_RPN.retain_grad()
                    if adv_s == adv_step - 1:
                            break
                    # delta's graduate
                    delta_RPN_grad = delta_RPN.grad.clone().detach()
                    # norm
                    denorm = torch.norm(delta_RPN_grad.view(delta_RPN_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta_RPN = (delta_RPN + adv_lr * delta_RPN_grad / denorm).detach()
                    # renew delta
                    delta_RPN_norm = torch.norm(delta_RPN.view(delta_RPN.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_RPN_norm > 1).to(RPN_embeddings)
                    reweights = (1 / delta_RPN_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta_RPN = (delta_RPN * reweights).detach()


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)
        #               Evaluation
        if evaluation:
            evaluate(model, dataloader_validation)