import os
import json
import numpy as np
from PIL import Image
import pickle

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from lshash import lshash
import matplotlib.pyplot as plt

class ImageEncoder(nn.Module):
    def __init__(self, modeltype):
        """Load the pretrained model and replace top fc layer."""
        super(ImageEncoder, self).__init__()
        if modeltype == 'resnet152':
            self.ImageEnc = models.resnet152(pretrained=True)
        elif modeltype == 'resnet101':
            self.ImageEnc = models.resnet101(pretrained=True)
        elif modeltype == 'resnet50':
            self.ImageEnc = models.resnet50(pretrained=True)
        elif modeltype == 'resnet18':
            self.ImageEnc = models.resnet18(pretrained=True)
        else:
            raise ValueError('{} not supported'.format(modeltype))
        self.layer = self.ImageEnc._modules.get('avgpool')
        self.ImageEnc.eval()


    def forward(self, images):
        """Extract the image feature vectors."""
        my_embedding = torch.zeros(1, 2048)
        def copy_data(m, i, o):
            my_embedding.copy_(torch.flatten(o.data, 1))
        h = self.layer.register_forward_hook(copy_data)
        self.ImageEnc(images)
        h.remove()

        return my_embedding

def extract_feats(input_image, cnnmodel):

    isvalid = False
    image_file = input_image

    try:
        image = Image.open(image_file).convert('RGB')
        image_loader = transforms.Compose(
                    [transforms.Resize(224),
                    transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])

        if image_loader(image).float().shape[0] == 1:
            image_vec = torch.from_numpy(
                    np.stack((image_loader(image).float())*3))
            print("Converted 1 channel image to 3 channel image.",
                  image_vec.shape)
        else:
            image_vec = image_loader(image).float()

        # image_vec = image_vec.cuda()
        featvec = cnnmodel(image_vec.unsqueeze(0)).cpu().numpy()
        isvalid = True
    except:
        featvec = None
        pass
    return featvec, isvalid


def get_similar_item(query_vec, lsh_variable, n_items=5):
    response = lsh_variable.query(query_vec, num_results=n_items+1, distance_func='hamming')

    top_k_sim_img = [response[i][0][1] for i in range(n_items)]

    return top_k_sim_img


def search(input_image_path):
    # For image embeddings
    cnnmodel = ImageEncoder('resnet152')
    lsh = lshash.LSHash(hash_size=10, input_dim=2048, num_hashtables=5)

    # Find similar items
    lsh = pickle.load(open('lsh.p', 'rb'))

    test_image_file = input_image_path
    query_vec, _ = extract_feats(test_image_file, cnnmodel)
    query_vec = query_vec.tolist()[0]
    similar_img_ids = get_similar_item(query_vec, lsh)

    return similar_img_ids

