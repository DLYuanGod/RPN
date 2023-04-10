import numpy as np
import torch

global selection_matrix

def selection_mat(batch_size,gpus,prob):
    batch_size = batch_size // gpus
    global selection_matrix
    selection_matrix = torch.from_numpy(np.random.choice([1., 0.], size=batch_size*50*768, p=[prob,1-prob]).reshape(batch,50,768)).cuda().to(torch.float32)

def Normal_noise(embeddings,input_mask):
    input_lengths = torch.sum(input_mask, 1)
    delta = torch.zeros_like(embeddings).uniform_(-1, 1)* input_mask.unsqueeze(2)
    dims = input_lengths * embeddings.size(-1)
    mag =  1 / torch.sqrt(dims)
    delta = (delta * mag.view(-1, 1, 1)).detach()
    return embeddings,delta

def Gaussian_noise(embeddings,input_mask,scale):
    dim1,dim2,dim3 = np.shape(embeddings)
    noise = np.random.normal(loc=0, scale=scale, size=(dim1,dim2,dim3))
    noise = torch.from_numpy(noise)
    noise = noise.to(embeddings) * input_mask.unsqueeze(2)
    enhanced_samples = noise + embeddings
    return enhanced_samples

#Add salt and pepper noise
def SAP_noise(embeddings,prob):
    embeddings = embeddings.cpu().detach().numpy()
    dim1,dim2,dim3 = np.shape(embeddings)
    #Set half to 0 and the other half to maximum
    S_selection_matrix = np.random.choice([False, True], size=dim1*dim2*dim3, p=[1-(prob/2.),prob/2.]).reshape(dim1,dim2,dim3)
    P_selection_matrix = np.random.choice([False, True], size=dim1*dim2*dim3, p=[1-(prob/2.),prob/2.]).reshape(dim1,dim2,dim3)
    max_value = embeddings.max()
    embeddings[S_selection_matrix] = 0.0
    embeddings[P_selection_matrix] = max_value
    embeddings = torch.from_numpy(embeddings).cuda().requires_grad_(True)
    return embeddings

#My Method Random Position Noise
def RPN_noise(embeddings):
    global selection_matrix
    copy_embeds = embeddings.clone()
    copy_embeds = selection_matrix * copy_embeds
    #Randomly coded
    idx = torch.randperm(copy_embeds.nelement())
    noise = copy_embeds.view(-1)[idx].view(copy_embeds.size())
    #The opposite selection matrix is also randomly coded
    selection_matrix = selection_matrix.view(-1)[idx].view(selection_matrix.size())
    #Set the corresponding position to 0
    embeddings = embeddings -  embeddings * selection_matrix
    enhanced_samples = (embeddings + noise).requires_grad_(True)
    return enhanced_samples