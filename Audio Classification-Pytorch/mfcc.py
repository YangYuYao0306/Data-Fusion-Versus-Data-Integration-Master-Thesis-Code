import os

import librosa
import numpy as np
from scipy.fftpack import dct
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt




def write_file(feats, file_name):
    f = open('./files/' + file_name, 'w')
    (row, col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i, j]) + ' ')
        f.write(']\n')
    f.close()


def plot_spectrogram(spec, x_name, y_name, file_name):
    fig = plt.figure(figsize=(20, 5))
    # Display different colours at different positions according to the values of the 2D matrix
    heatmap = plt.pcolor(spec)
    # Colour bar showing the amplitude corresponding to the depth of the colour
    fig.colorbar(mappable=heatmap)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig('./pics/' + file_name)


'''
function:Creating 2D signal images
input:@signal_nameSignal Name,@x_nameName X-axis,@y_nameName Y-axis,@save_nameSave image name,@x-axis X-value,@y-axis Y-value
    @mode is the mode, if it is 1, the X-axis scale is clear.
output:None
'''


def Creat_Image(signal_name, x_name, y_name, save_name, x, y, mode=0):
    fig = matplotlib.pyplot
    fig.figure(figsize=(15, 5))
    if mode == 1:
        fig.xticks([])
    fig.title(signal_name)
    fig.xlabel(x_name)
    fig.ylabel(y_name)
    fig.plot(x, y)
    fig.savefig('./pics/' + save_name)


'''
function:Pre-emphasis of the audio signal to increase the energy of the high-frequency part of the signal.
input:@signal input one-dimensional array of signals to be pre-emphasised,@coeff pre-emphasis coefficients
output:Pre-emphasised one-dimensional signal np array
'''


def preemphasis(signal, coeff=0.7):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


'''
function:Pre-emphasis frame-wise windowing to reduce energy leakage after FFT
input:@signal input signal @frame_len frame length @frame_shift frame shift @win window function
output: A two-dimensional array of frames and windows, with the two dimensions being the signal of a frame and the number of frames respectively
'''


def enframe(signal, frame_len, frame_shift, win):
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames


'''
function:fft for each frame
input:@frames 2D array of frames @fft_len frame length
output: 2D array of signals after FFT of each frame, possibly complex amplitude, [number of signal frames, number of valid frequency components per frame].
'''


def get_spectrum(frames, fft_len):
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2)
    spectrum = np.abs(cFFT[:, 0:(valid_len + 1)])
    return spectrum


'''
function:for each frame after mel filter and take log operation
input:@spectrum input spectrum after fft [number of signal frames, number of valid frequency components per frame] @num_filter number of filters
output: 2D array of the output from the logging operation on each frame after the Meier filter [number of signal frames, number of filters].
'''


def fbank(spectrum, num_filter):
    low_freq_mel = 0  # Minimum Mel Frequency
    '''The maximum frequency of the discrete Fourier transform is the period of the delayed signal 1/T, batch is k/NT,
    Calculate the maximum value of the Meier frequency using the formula,# because the spectrogram of the discrete Fourier transform is conjugate symmetric so only the first half of the spectrum is useful.
    Because the spectrum of the discrete Fourier transform is conjugate symmetric, only the first half of the spectrum is useful.
    '''
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # The number of Mel filters divided into a specified number of parts equally spaced between the highest and lowest mel frequencies.
    # To facilitate the later calculation of the mel filter bank, the left and right sides are complemented with a centre point each
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # Inverse formula to calculate the linear frequency corresponding to the Meier frequency
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    # nfilt dimensional array holding spectral data for each Meier filter [number of filters, number of valid frequency components per frame]
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    # Each centre frequency corresponds to the index of the first Mel filter
    bin = np.floor(hz_points * (NFFT / 2) / (sample_rate / 2))

    # Since two centre points are made up before and after, the first filter is centred at bin[1],and the last at bin[nfilt+1].
    # m is the data of the first filter, and k is the first point of it.
    for m in range(1, nfilt + 1):
        left = int(bin[m - 1])
        center = int(bin[m])
        right = int(bin[m + 1])
        for k in range(left, center):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(center, right):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    pow_frames = spectrum  # spectral energy
    # Mel filter matrix multiplied by the transpose of the input signal matrix to obtain the fbank data for each frame
    # Dimension is [number of signal frames, number of filters]
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Data Security
    # Taking log,unit db, is to facilitate the MFCC feature extraction to remultiply into an additive
    filter_banks = 20 * np.log10(filter_banks)
    feats = filter_banks
    return feats


# MFCC feature extraction to obtain spectral envelope and spectral details MFCC (Mel Frequency Cepstrum Coefficient)
'''
function:@mfcc feature extraction for fbank after logging
input:@fbank for each frame after mel filter and log operation output 2D array [number of signal frames, number of filters] @num_mfcc take the dimension of envelope information
output: 12-dimensional MFCC features [number of signal frames, dimension of envelope information].
'''


def mfcc(fbank, num_mfcc):
    # If you don't make the discrete spectrum symmetric, you can use DCT to get the cepstrum coefficient of Fbank.
    # After taking the log, the envelope and the details become additive, the frequency of the envelope is lower and the first k spectral densities are taken as the characteristics of the envelope.
    mfcc = dct(fbank, type=2, axis=1, norm='ortho')[:, 1: (num_mfcc + 1)]
    feats = mfcc
    return feats


def main(o,wav,sample_rate,frame_len,frame_shift,preemphasis_coeff):
    os.makedirs('./pics/'+o.replace('.',''))
    Creat_Image('wav_signal', 'Time', 'Value', o.replace('.','')+'/Initial audio signal.png', np.arange(0, len(wav)) / sampling_rate, wav, 0)
    signal = preemphasis(wav,preemphasis_coeff)
    Creat_Image('preemphasis_signal', 'Time', 'Value', o.replace('.','')+'/pre-emphasis signal.png', np.arange(0, len(signal)) / sampling_rate, signal,
                0)
    frames = enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len))
    Creat_Image('dev_cut_signal', 'Time', 'Value', o.replace('.','')+'/Expand after adding window by frame.png', np.arange(0, (frames.shape[0]) * (frames.shape[1])),
                frames.flatten(), 1)
    spectrum = get_spectrum(frames,fft_len)
    Creat_Image('spectrum_signal', 'Frequency', 'Value', o.replace('.','')+'/Spectrum of signals per frame.png',
                np.arange(0, (spectrum.shape[0]) * (spectrum.shape[1])), spectrum.T.flatten(), 1)
    fbank_feats = fbank(spectrum,num_filter)
    mfcc_feats = mfcc(fbank_feats,num_mfcc)

    plot_spectrogram(fbank_feats.T, 'Frames', 'Filter Bank', o.replace('.','')+'/fbank.png')
    write_file(fbank_feats, './test.fbank')
    plot_spectrogram(mfcc_feats.T, 'pass', 'MFCC', o.replace('.','')+'/mfcc.png')
    write_file(mfcc_feats, './test.mfcc')
    print("operate successful!")


# if __name__ == "__main__":
    # sampling rate
sampling_rate = 16000
# Read the audio signal stored in a one-dimensional array, fs is the sampling frequency, sr = None is the original sampling rate

for o in os.listdir('./dataset/audio/all'):
    path = os.path.join('./dataset/audio/all',o)

    wav, fs = librosa.load(path, sr=sampling_rate)
    sample_rate = fs
    # Pre-emphasis factor
    preemphasis_coeff = 0.97
    # Frame length, generally 33 to 100 frames per second, take 40 frames, frame period 25ms in accordance with the sampling law
    frame_len = int(sampling_rate / 40)
    # Frame shift, take 0.25 for frame shift/frame length
    frame_shift = int(frame_len * 0.25)
    # N-point discrete Fourier transform, spectral conjugate symmetry, first half effective
    fft_len = 512
    NFFT = fft_len
    # Number of Mel filters
    num_filter = 23
    nfilt = num_filter
    # Taking the first 12 dimensions of the MFCC as spectral envelope features
    num_mfcc = 12

    main(o,wav,sample_rate,frame_len,frame_shift,preemphasis_coeff)
