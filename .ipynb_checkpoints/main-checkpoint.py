from statistics import mode
import torch
import argparse
from transformers import XLNetTokenizer
from transformers import RobertaTokenizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os

import building_the_dataset as bd
import encoded_data as encoded
import trainer
import add_noise as an
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
def tokenizer_select(model):
    if model == 'XLNet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif model == 'Roberta':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def building_tensordataset(inputs_id,mask,label):
    return (TensorDataset(inputs_id,mask,label))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp','--train_data_path',default='.', type=str,
                        help="which data path")
    parser.add_argument('-vp','--val_data_path',default='.', type=str,
                        help="which data path")
    parser.add_argument('-tep','--test_data_path',default='.', type=str,
                        help="which data path")
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help="which gpu to have")
    parser.add_argument('-e', '--epochs', default=3, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-lr', '--lr', default=5e-6, type=float, 
                        metavar='Ne-N',
                        help='model value of learning rate')
    parser.add_argument('-eps', '--eps', default=1e-8, type=float, 
                        metavar='Ne-N',
                        help='model value of precision')
    parser.add_argument('-b', '--batch_size', default=128, type=int, 
                        metavar='N',
                        help='number of batchsize')
    parser.add_argument('-m','--model',default='XLNet', type=str,
                        help="which model use XLNet or Roberta")
    parser.add_argument('-md','--mode',default='test', type=str,
                        help="which mode use test or val")
    parser.add_argument('-ns','--noise',default='None', type=str,
                        help="which noise use None ,Gaussian(G) ,SAP ,RPN or whole")
    parser.add_argument('-sc','--scale',default='0.01', type=float,
                        help="Variance of Gaussian distribution")
    parser.add_argument('-pr','--prob',default='0.0', type=float,
                        help="SAP and RPN selection probability")
    parser.add_argument('-as', '--adv_step', default=1, type=int, 
                        metavar='N',
                        help='number of adv step')
    parser.add_argument('-tk','--tasks_kinds',default='None', type=str,
                        help="which tasks_kinds use Classification or MultipleChoice")
    parser.add_argument('-nl','--num_labels',default=1, type=int,
                        help="If you use Classification.Please input num_labels")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()


    print("Data cleaning in progress!\n")
    train_data_text,train_data_label = bd.building_dataset(args.train_data_path)
    val_data_text,val_data_label     = bd.building_dataset(args.val_data_path)
    test_data_text,test_data_label   = bd.building_dataset(args.test_data_path)

    print('Sample word segmentation and coding in progress!\n')
    tokenizer = tokenizer_select(args.model)
    encoded_data_train =  encoded.encoded_data(tokenizer,train_data_text)
    encoded_data_val   =  encoded.encoded_data(tokenizer,val_data_text)
    encoded_data_test  =  encoded.encoded_data(tokenizer,test_data_text)

    print('Dataset construction in progress!\n')
    dataset_train = building_tensordataset(encoded_data_train['input_ids'], encoded_data_train['attention_mask'], torch.Tensor(train_data_label).long())
    dataset_val = building_tensordataset(encoded_data_val['input_ids'], encoded_data_val['attention_mask'], torch.Tensor(val_data_label).long())
    dataset_test = building_tensordataset(encoded_data_test['input_ids'], encoded_data_test['attention_mask'], torch.Tensor(test_data_label).long())

    dataloader_train = trainer.building_dataloader(dataset_train,args.batch_size,args.gpus,mode='train')
    dataloader_val = trainer.building_dataloader(dataset_val,args.batch_size,args.gpus,mode='else')
    dataloader_test = trainer.building_dataloader(dataset_test,args.batch_size,args.gpus,mode='else')
    print('Instantiating model!\n')
    model, optimizer, scheduler = trainer.initialize_model(args.model,args.gpus,args.local_rank,args.lr,args.eps,dataloader_train,epochs=args.epochs,tasks_kinds=args.tasks_kinds,num_labels=args.num_labels)

    if args.mode == 'test':
        if args.noise == 'RPN':
            an.selection_mat(args.batch_size,args.gpus,args.prob)
        print(f"{'Epoch':^7} | {'Every 40 Batchs':^9} | {'Train dataset Loss':^12} | {'test/val dataset Loss':^10} | {'test/val acc':^9} | {'time':^9}")
        trainer.train(model=model,train_dataloader=dataloader_train,optimizer=optimizer,scheduler=scheduler,epochs=args.epochs
            ,noise=args.noise,scale=args.scale,prob=args.prob, evaluation=False,gpus=args.gpus,adv_step=args.adv_step)
        trainer.evaluate(model,dataloader_test)


    elif args.mode == 'val':
        if args.noise == 'RPN':
            an.selection_mat(args.batch_size,args.gpus,args.prob)
        print(f"{'Epoch':^7} | {'Every 40 Batchs':^9} | {'Train dataset Loss':^12} | {'test/val dataset Loss':^10} | {'test/val acc':^9} | {'time':^9}")
        trainer.train(model=model,train_dataloader=dataloader_train,optimizer=optimizer,scheduler=scheduler,epochs=args.epochs
            ,noise=args.noise,scale=args.scale,prob=args.prob, evaluation=False,gpus=args.gpus,adv_step=args.adv_step)
        val_loss, val_accuracy = trainer.evaluate(model,dataloader_val)
        print(args.mode + 'loss:' + str(val_loss) +'||' + 'acc:' + str(val_accuracy))
        val_loss, val_accuracy = trainer.evaluate(model,dataloader_test)
        print(args.mode + 'loss:' + str(val_loss) +'||' + 'acc:' + str(val_accuracy))


if __name__ == '__main__':
    main()