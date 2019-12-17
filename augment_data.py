import json
import random
from sklearn.utils import shuffle
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import torch
import pdb

import argparse
config = json.load(open('config.json', 'r'))   

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
    
def extract_feats(img_id, cnnmodel):
        
    isvalid = False
    image_file = os.path.join(config['image_data_dir'], img_id + '.jpeg')
#    if os.path.getsize(image_file)/1024 < 30:
#        return None, isvalid
    try:
        image = Image.open(image_file).convert('RGB')
        image_loader = transforms.Compose(
                    [transforms.Resize(config["input_image_dim"]),
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
 
def get_personality_descriptive(filename):
    descriptive_captions = {}
    with open('./data/personality_captions/' + 
              'personality_captions_image_MSCOCO_caption.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').strip('\r')
            image_file, caption = line.split('\t')
            image = image_file.split('.')[0]
            if image not in descriptive_captions:
                descriptive_captions[image] = caption
    f.close() 
     
    caption_personality = {}
    with open('./data/personality_captions/' + filename + '.json', 'r') as f:
        data = json.load(f)
        for doc in data:
            personality = doc['personality']
            caption = doc['comment']
            if personality not in caption_personality:
                caption_personality[personality] = []
            caption_personality[personality].append(caption)
    f.close()
    
    image_personality = {}
    image_caption = {}
    with open('./data/personality_captions/' + filename + '.json', 'r') as f:
        data = json.load(f)
        for doc in data:
            image_hash = doc['image_hash']
            personality = doc['personality']
            caption = doc['comment']
            image_personality[image_hash] = personality
            image_caption[image_hash] = caption
    f.close()
    
    return caption_personality, descriptive_captions, image_personality, image_caption
 
def get_data(filename, numsamples, caption_personality, 
             descriptive_captions, removedimages, image_personality, 
             image_caption, cluster_number_images, image_cluster_number):
    output_data = []
    with open('./data/personality_captions/' + filename + '.json', 'r') as f:
        data = json.load(f)
        # shuffle data and keep only k data points (for faster training)
        data = shuffle(data)
        count = 0
        totcount = len(data)
        removecount = 0
        for doc in data:
            image_hash = doc['image_hash']
            # do not consider samples with broken images
            if image_hash in removedimages:
                removecount += 1
                continue
            # Get the descriptive MS-COCO caption
            desc_caption = ''
            if image_hash in descriptive_captions:
                desc_caption = descriptive_captions[image_hash]
            else:
                print('Descriptive caption not found for image ', image_hash)
            doc['descriptive'] = desc_caption
            
            '''
            Get negative abstract captions
            
            - images in each split (train, test, val) have been clustered with similarity 0.6
            - key 'unpaired_abstract_same_per': abstract caption from a different cluster having same personality
            - key 'unpaired_abstract_diff_per': abstract caption from a different cluster having different personality
            - each of the above keys have at most 50 captions
            
            '''
            cluster_number = 0
            # some images cannot be put in clusters
            if image_hash in image_cluster_number:
                cluster_number = image_cluster_number[image_hash]
            
            personality = doc['personality']
            
            # randomly sample captions from different clusters
            available_clusters = set(cluster_number_images.keys()) - set(
                    [cluster_number])
                        
            unpaired_same_personality = [] # list of unpaired captions adhering to the same personality
            unpaired_diff_personality = [] # list of unpaired captions of different personality
            counter_same_personality = 0
            counter_diff_personality = 0
            number_unpaired_captions = 50
            
            while (counter_same_personality < number_unpaired_captions) or (
                    counter_diff_personality < number_unpaired_captions):
                if len(available_clusters) == 0:
                    break
                random_cluster = random.sample(available_clusters, 1)[0]
                for random_image in cluster_number_images[random_cluster]:
                    unpaired_caption = image_caption[random_image].replace(
                    ',', ' ,').replace('.', ' .').replace(
                            ':', ' :').replace(';', ' ;').replace(
                                    '?', ' ?').replace('!', ' !')
                    # randomly sample captions of the same personality
                    if image_personality[random_image] == personality:
                        if counter_same_personality < number_unpaired_captions:
                            unpaired_same_personality.append(unpaired_caption)
                            counter_same_personality += 1
                    # randomly sample captions of different personality
                    else:
                        if counter_diff_personality < number_unpaired_captions:
                            unpaired_diff_personality.append(unpaired_caption)
                            counter_diff_personality += 1
                            
                available_clusters = set(available_clusters) - set(
                        [random_cluster]) # remove the cluster already seen
            
            doc['unpaired_abstract_same_per'] = unpaired_same_personality
            doc['unpaired_abstract_diff_per'] = unpaired_diff_personality
                
            
            # Space before punctuations for positive and negative abstract captions
            doc['comment'] = doc['comment'].replace(
                    ',', ' ,').replace('.', ' .').replace(
                            ':', ' :').replace(';', ' ;').replace(
                                    '?', ' ?').replace('!', ' !')
            
            # removing 'neg_abstract' since it is now redundant
            if 'neg_abstract' in doc:
                del doc['neg_abstract']
            
            if desc_caption != '':
                output_data.append(doc)
                count += 1
            if numsamples is not None:
                if count == numsamples:
                    break
            print('num samples processed: {}/{}'.format(count, totcount), 
                  end='\r')
        print('\n no of samples removed: {}'.format(removecount))
    return output_data  


def read_clusters(split):
    with open('./data/personality_captions/image_clusters_' + split + 
              '.txt', 'r') as f:
        lines = f.readlines()
        cluster_number_images = {}
        image_cluster_number = {}
        for line in lines:
            line = line.rstrip().rstrip(',')
            cluster_number = line.split("\t")[0]
            images = line.split("\t")[1].split(",")
            cluster_number_images[cluster_number] = images
            for image in images:
                image_cluster_number[image] = cluster_number
        return cluster_number_images, image_cluster_number
            
def main_func(flags):
     
    if flags.num_samples == 10000:
        num_samples = [10000, 100, 100]
    elif flags.num_samples == 10:
        num_samples = [10, 5, 5]
    elif flags.num_samples == 0:
        num_samples = [None, None, None]
        
#    process = ['train', 'val', 'test']
      
    if flags.extract_feat:
        # init image features stuff
        modeltype = flags.modeltype
        imgfeatsavedir = config['image_feat_data_dir'] + '/' + modeltype + "/"
        cnnmodel = ImageEncoder(modeltype)
#        cnnmodel = cnnmodel.cuda()
        print('image encoder {} loaded..'.format(modeltype))
          
        if not os.path.exists(imgfeatsavedir):
            os.makedirs(imgfeatsavedir)
          
        # Extract features and store images that caused errors
        allimages = set()
        removedimages = set()
        
        print(os.getcwd())
        input_files = os.listdir(config['image_data_dir'])
        print(input_files)
      
#        for idx in range(len(process)):
#
#            proc = process[idx]
#            print('extracting features for {} set..'.format(proc))
#            with open('./data/personality_captions/' + proc + '.json', 'r') as f:
#                data = json.load(f)
        tempcnt = 0
#                totcnt = len(data)
        for file in input_files:
            imagehash = file.split('.')[0]
            if imagehash not in allimages and imagehash not in removedimages:
                  
                output_feat, isvalid = extract_feats(imagehash,
                                                     cnnmodel)
                  
                if isvalid:
                    allimages.add(imagehash)
                    tempcnt += 1
                    np.save(imgfeatsavedir + imagehash + '.npy',
                            output_feat)
                else:
                    removedimages.add(imagehash)
  
                print('Valid images: {}, invalid images: {}'.format(
                        len(allimages), len(removedimages)), end='\r')
            else:
                pass

        print('Total no of Valid images: {}, invalid images: {}'.format(
                len(allimages), len(removedimages)))
          
        json.dump(list(removedimages), 
                  open('./data/removed_images.json', 'w'))
    else:
          
        removedimages = json.load(
                open('./data/removed_images.json', 'r'))

    # prepare data
#    for idx in range(len(process)):
#
#        proc = process[idx]
#        numsamp = num_samples[idx]
#        if numsamp is not None:
#            op_file = proc + '_' + str(numsamp)
#            ip_file = proc
#        else:
#            op_file = proc + '_processed'
#            ip_file = proc
#
#        # read image clusters
#        cluster_number_images, image_cluster_number = read_clusters(proc)
#
#        # get descriptive captions and personalities from train data.
#        (per_captions, desc_captions, image_personality,
#         image_caption) = get_personality_descriptive(ip_file)
#
#        fout = open('./data/personality_captions/' + op_file + '.json', 'w+')
#
#        output_data = get_data(ip_file, numsamp, per_captions, desc_captions,
#                               removedimages, image_personality, image_caption,
#                               cluster_number_images, image_cluster_number)
#
#        json.dump(output_data, fout)
#        print('data for {} extracted: {} samples'.format(proc,
#              len(output_data)))
    
    return True
     
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aumgent data parameters')
    parser.add_argument('--extract_feat', action='store_true', 
                        help='do we need to extract features for all images?')
    parser.add_argument('--modeltype', default='resnet152', type=str,
                        help='which model for image features?')
    parser.add_argument('--process_data', action='store_true', 
                        help='are we using bert to get word vectors?')
    parser.add_argument('--num_samples', default=0, type=int)
    
    flags = parser.parse_args()
    main_func(flags)
         
    
