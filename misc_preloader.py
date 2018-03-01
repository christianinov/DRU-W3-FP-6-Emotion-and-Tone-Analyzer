import numpy as np
import tflearn
from scipy.io import wavfile
from scipy import signal
import librosa as lr

def misc_preloader(target_path,
                  normalize=False,
                  resample=None, 
                  crop=None,
                  categorical_labels=True,
                  remove_silence_threshold=None):
    with open(target_path, 'r') as f:
        wavs, labels = [], []
        for l in f.readlines():
            l = l.strip('\n').split()
            wavs.append(l[0])
            labels.append(int(l[1]))

    n_classes = np.max(labels) + 1
    X = MiscPreloader(wavs, normalize, resample, crop, remove_silence_threshold)
    Y = tflearn.data_utils.LabelPreloader(labels, n_classes, categorical_labels)
    return X, Y

class MiscPreloader(tflearn.data_utils.Preloader):
    
    def __init__(self, array, normalize, resample, crop, remove_silence_threshold):
        fn = lambda x: self.preload(x, normalize, resample, crop, remove_silence_threshold)
        super(MiscPreloader, self).__init__(array, fn)
    
    @staticmethod
    def remove_silence_func(sample, err):
        start = 0
        finish = len(sample) - 1
        while np.abs(sample[start]) < err:
            start += 1
        while np.abs(sample[finish]) < err:
            finish -= 1
        return np.copy(sample[start:finish])
    
    @staticmethod
    def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)
    
    @staticmethod
    def mp3_to_img(path, height=192, width=192):
        signal, sr = lr.load(path, res_type='kaiser_fast')
        #hl = signal.shape[0]//(width*1.1) #this will cut away 5% from start and end
        hl = signal.shape[0]//width
        spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
        img = lr.logamplitude(spec)**2
        start = (img.shape[1] - width) // 2
        img = img[:, start:start+width]
        img /= np.max(img)
        return img

    def preload(self, path, normalize, resample, crop, remove_silence_threshold):
        return MiscPreloader.mp3_to_img(path)
        
    """def preload(self, path, normalize, resample, crop, remove_silence_threshold):
        sample_rate, sample = wavfile.read(path)
        if resample is not None:
            sample = signal.resample(sample, int(resample/sample_rate * sample.shape[0]))
        if remove_silence_threshold is not None:
            sample = SpectrogramPreloader.remove_silence_func(sample, remove_silence_threshold)
        if normalize:
            sample = sample * 1.0 / np.amax(sample)
        if crop is not None:
            sample = sample[-crop:]
        freqs, times, spectrogram = SpectrogramPreloader.log_specgram(sample, sample_rate)
        spectrogram = spectrogram.T
        left = int((spectrogram.shape[1] - SpectrogramPreloader.spect_shape[1]) / 2)
        right = int(spectrogram.shape[1] - SpectrogramPreloader.spect_shape[1] - left)
        return spectrogram[:,left:-right]"""