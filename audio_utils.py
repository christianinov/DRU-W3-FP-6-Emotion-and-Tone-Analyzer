from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np

class Sample:
    
    def __init__(self, filename):
        self.sig = AudioSegment.from_file(filename)
        
    def get_bit_rate(self):
        return self.sig.sample_width
    
    def get_sample_rate(self):
        return self.sig.frame_rate
    
    def get_seconds(self):
        return self.sig.duration_seconds
    
    def get_data(self):
        return np.array(self.sig.get_array_of_samples())
        
    def resample(self, new_bit_rate = 2, new_frame_rate = 22050):
        self.sig = self.sig.set_sample_width(new_bit_rate)
        self.sig = self.sig.set_frame_rate(new_frame_rate)
        
    def remove_silence(self, min_silence_len=10, silence_thresh=-100, keep_silence=10):
        parts = split_on_silence(self.sig, min_silence_len, silence_thresh, keep_silence)
        res = AudioSegment.empty()
        for part in parts:
            res += part
        self.sig = res