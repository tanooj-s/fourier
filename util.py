from scipy import fftpack
import numpy as np
import librosa.core as lc
import librosa.feature as lf
import librosa.decompose
import librosa.effects as le

def load_sample(path,upper_bound,lower_bound):
	samples, rate = lc.load(path=path, mono=True, duration=upper_bound) 
	samples, _ = le.trim(samples[int(rate*lower_bound):int(rate*upper_bound)], top_db=20)
	return samples, rate


def get_fft(samples,rate):
	nyquist = int(rate/2)
	yfft = fftpack.fft(x=samples,n=rate)
	max_amp = np.max(np.abs(yfft))
	xfft = np.linspace(0,nyquist,nyquist)
	yfft = np.abs(yfft[:nyquist])/(max_amp) 
	return {'x': xfft, 'y': yfft}

def get_peak_frequencies(lst): 
	# for this use case lst is assumed to be fft['y'] 
	indices = sorted(np.argpartition(lst,-50)[-50:]) 
	peak_freqs = []
	for i in indices:
		if lst[i] > 0.3:
			peak_freqs.append(i)
	peak_freqs = sorted(purge(peak_freqs)) # only retain mean of clusters
	peak_freqs = [[str(p), lc.hz_to_note(p)] for p in peak_freqs] 
	return peak_freqs


def purge(lst):
	# from a list of numbers, if there are any clusters, get the mean of the clusters 
	lst = sorted(lst)
	stack = []
	out = []
	threshold = 5 # should be adjustable, possibly on a log scale
	for num in lst:
		if len(stack) == 0:
			stack.append(num)
		elif (num - stack[-1]) < threshold: 
			stack.append(num)
		else:
			out.append(np.mean(stack))
			stack = []
			stack.append(num)
	if len(stack) > 0:
		out.append(np.mean(stack))
	return out	


def get_chord(note_list):
	if len(note_list) < 3:
		return 'Not enough notes for a chord'
	else:
		chord_hash = ''
		for n in note_list:
			chord_hash = chord_hash + str(note_to_ix[n[:-1].lower()])
		for k in chord_map.keys():
			if k in chord_hash:
				return chord_map[k]
		return 'Unidentified chord'


def extract_features(signal): 
	return [librosa.feature.zero_crossing_rate(signal)[0], librosa.feature.spectral_centroid(signal)[0]]

note_to_ix = {
	'c': '1',
	'c#': '2',
	'd': '3',
	'd#': '4',
	'e': '5',
	'f': '6',
	'f#': '7',
	'g': '8',
	'g#': '9',
	'a': '10',
	'a#': '11',
	'b': '12'
}

# built out from note_to_ix above, to avoid issues with sharp characters when sorting
chord_map = {
	'158': 'C Major',
	'3710': 'D Major',
	'5912': 'E Major',
	'6101': 'F Major',
	'8123': 'G Major',
	'1025': 'A Major',
	'1247': 'B Major',

	'148': 'C Minor',
	'3610': 'D Minor',
	'5812': 'E Minor',
	'691': 'F Minor',
	'8113': 'G Minor',
	'1015': 'A Minor',
	'1237': 'B Minor'
}


cmap_opts = [
			'Blues', 'BrBG', 'BuGn', 'BuPu', 'GnBu', 'Greens', 
			'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 
			'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 
			'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 
			'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 
			'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 
			'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 
			'gnuplot', 'gnuplot2', 'gray', 'hot',  
			'seismic', 'spring', 'summer', 
			'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 
			'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Blues_r', 
			'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r', 'Greys_r', 
			'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 
			'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 
			'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 
			'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 
			'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 
			'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 
			'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 
			'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 
			'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 
			'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 
			'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r', 'magma', 'magma_r', 
			'inferno', 'inferno_r', 'plasma', 'plasma_r', 'viridis', 'viridis_r', 
			'cividis', 'cividis_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r']

