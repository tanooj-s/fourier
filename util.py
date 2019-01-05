from scipy import fftpack
import numpy as np
import librosa.core as lc
def get_fft(samples,rate):
	nyquist = int(rate/2)
	yfft = fftpack.fft(x=samples,n=rate)
	max_amp = np.max(np.abs(yfft))
	xfft = np.linspace(0,nyquist,nyquist)
	yfft = np.abs(yfft[:nyquist])/(max_amp) 
	return {'x': xfft, 'y': yfft}

def get_peak_frequencies(lst):
	# for this use case lst is assumed to be fft['y'] i.e. the relative amplitudes of the fourier tranform
	indices = sorted(np.argpartition(lst,-15)[-15:]) # get indices of 15 max values in the list (these essentially correspond to fft['x'])
	peak_freqs = []
	for i in indices:
		if lst[i] > 0.3:
			peak_freqs.append(i)
	peak_freqs = purge(peak_freqs) # only retain mean of clusters
	peak_freqs = [[str(p),lc.hz_to_note(p)] for p in peak_freqs]
	return peak_freqs

def purge(lst):
	# from a list of numbers, if there are any clusters, get the mean of the clusters
	lst = sorted(lst)
	stack = []
	out = []
	for num in lst:
		if len(stack) == 0:
			stack.append(num)
		elif (num - stack[-1]) < 5: # peak frequency "cluster" threshold
			stack.append(num)
		else:
			out.append(np.mean(stack))
			stack = []
			stack.append(num)
	if len(stack) > 0:
		out.append(np.mean(stack))
	return out	