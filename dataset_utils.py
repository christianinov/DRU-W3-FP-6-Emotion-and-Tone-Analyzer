import numpy as np
import os
# from scipy.io import wavfile
from audio_utils import Sample

class Dataset:

    def __init__(self, directory, feature_extraction_function):
        self.directory = directory
        self.feature_extraction_function = feature_extraction_function
        self.__create_categories(directory)
        self.__create_dataset()
        
    def __create_categories(self, directory):
        categories_folders = [name for name in os.listdir(directory) if os.path.isdir(directory+name)]
        self.categories = {}
        category_N = 0
        for category in categories_folders:
            self.categories[category_N] = category
            category_N += 1       
        
    def __create_dataset(self):
        self.dataset = []
        for category_N, category in self.categories.items():
            waves = [f for f in os.listdir(self.directory + category) if f.endswith('.wav')]
            for wav in waves:
                sample = Sample(self.directory + category + '/' + wav)
                sample.resample()
                sample.remove_silence()
                rate, sig = sample.get_sample_rate(), sample.get_data()
                # try:

                #     rate, sig = wavfile.read(self.directory + category + '/' + wav)
                # except ValueError:
                #     print('Bad sample: ' + self.directory + category + '/' + wav)
                #     continue
                mfcc_mean, mfcc_std, logfbank_mean, logfbank_std = self.feature_extraction_function(sig, rate)
                features = np.concatenate((mfcc_mean, mfcc_std, logfbank_mean, logfbank_std, [category_N]))
                self.dataset.append(features)
        self.dataset = np.asarray(self.dataset)
        
    def get_categories(self):
        return self.categories
    
    def get_dataset(self):
        return self.dataset
    
    def extend_with_dataset(self, other_dataset):
        raise Exception('Not implemented yet')
        
    def save_to_csv(self, filename):
        np.savetxt(filename, self.dataset, delimiter=",")
       
    def read_from_csv(self):
        raise Exception('Not implemented yet')
