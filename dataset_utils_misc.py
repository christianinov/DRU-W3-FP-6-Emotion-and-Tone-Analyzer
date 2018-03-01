import os
from sklearn.model_selection import train_test_split

# todo: module for cheching if sample is good or bad
from scipy.io import wavfile
import librosa as lr
import numpy as np

def get_categories(directory):
    categories = [name for name in os.listdir(directory) 
              if os.path.isdir(directory+name)
             and not name.count('dataset-txt')]
    return categories

def create_dataset_lists(directory):  
    
    if not os.path.exists(directory+'/dataset-txt'):
        os.makedirs(directory+'/dataset-txt')

    dataset_all = directory+'/dataset-txt/dataset-all.txt'
    dataset_train = directory+'/dataset-txt/dataset-train.txt'
    dataset_test = directory+'/dataset-txt/dataset-test.txt'
    dataset_validation = directory+'/dataset-txt/dataset-validation.txt'

    dataset = []

    categories = get_categories(directory)
    
    category_mapping = {}
    for i in range(len(categories)):
        category_mapping[categories[i]] = i

    for category in categories:
        for sample in os.listdir(directory+category):
            path = directory+category+'/'+sample
            #try:
                #signal, sr = lr.load(path, res_type='kaiser_fast')
                #hl = signal.shape[0]//(width*1.1) #this will cut away 5% from start and end
                #hl = signal.shape[0]
                #height = 192
                #width = 192
                #spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
                #img = lr.logamplitude(spec)**2
                #start = (img.shape[1] - width) // 2
                #img = img[:, start:start+width]
            #except (ValueError, lr.util.exceptions.ParameterError):
                #print ('Bad sample', path)
                #continue
            label = category_mapping[category]
            dataset.append((path,label))

    train, test = train_test_split(dataset, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)

    with open(dataset_all,'w') as f:
        for (path, label) in dataset:
            f.write('{} {}\n'.format(path,label))
    with open(dataset_train,'w') as f:
        for (path, label) in train:
            f.write('{} {}\n'.format(path,label))                    
    with open(dataset_test,'w') as f:
        for (path, label) in test:
            f.write('{} {}\n'.format(path,label))                    
    with open(dataset_validation,'w') as f:
        for (path, label) in val:
            f.write('{} {}\n'.format(path,label)) 