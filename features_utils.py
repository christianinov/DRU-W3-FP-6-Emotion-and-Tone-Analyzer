from python_speech_features import mfcc

def features_extraction(sig, rate, numcep=26, nfft=2048, **kwargs):
    """Compute mean and standard deviation of each MFCCs from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 2048.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: Two numpy arrays of size (numcep,). First vector contains mean of MFCCs, second - standard deviation of MFCCs
    """
    mfcc_feat = mfcc(sig, rate, numcep=numcep, nfft=nfft, **kwargs)
    mfcc_mean = mfcc_feat.mean(axis=0)
    mfcc_std = mfcc_feat.std(axis=0)
    return mfcc_mean, mfcc_std