
from flask import url_for, redirect, render_template, Flask, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SelectMultipleField, BooleanField, widgets, RadioField, SubmitField
from flask_wtf.file import FileField
from werkzeug import secure_filename

import time
import numpy as np
import matplotlib.pyplot as plt

import librosa.core as lc
from librosa import amplitude_to_db
import librosa.display as ld
import librosa.effects
import librosa.decompose
from librosa.feature import chroma_stft, chroma_cens, chroma_cqt, spectral_centroid, mfcc, zero_crossing_rate
from librosa.output import write_wav

from scipy import fftpack
import sklearn
from util import get_fft, purge, get_peak_frequencies, get_chord, note_to_ix, chord_map, extract_features, get_components, load_sample


class UploadForm(FlaskForm):
	user_file = FileField(label='User file')
	lower_bound_time = FloatField(label='Lower timestamp: ')
	upper_bound_time = FloatField(label='Upper timestamp: ')
	n_components = FloatField(label='Number of components: ')
	spec_opts = RadioField(label='Spectrogram options',coerce=int,choices=[(1,'STFT'),(2,'Linear Scale STFT'),(3,'CQT'),(4,'Chromagram')]) # choose better options
	d_opts = RadioField(label='Decomposition techniques',coerce=int,choices=[(1,'NMF (default)'),(2,'Dictionary Learning'),(3,'PCA'),(4,'Fast ICA') # sparse coding with a dictionary? might be that different NMF techniques are the best])
	
app = Flask(__name__)
app.config['SECRET_KEY'] = 'AHs0JleAhvzcolWG'
app.config['UPLOADED_FILES_DEST'] = 'uploads/'

dco_to_title = {1:'Nonnegative Matrix Factorization', 2:'Minibatch Dictionary Learning', 3:'Principal Conponent Analysis', 4:'Fast ICA'}
cmap = plt.get_cmap('winter')

# homepage 
@app.route('/',methods=['GET','POST'])
def upload():

	form = UploadForm()
	if form.validate_on_submit(): 
		flash("Successfully uploaded file!")
		curr_time = str(time.time())[:10] # get epoch time without decimals to save to files later
		session['file_name'] = secure_filename(form.user_file.data.filename) # why is it saying form.user_file.data is None before I even have the chance to fill it out
		form.user_file.data.save('uploads/' + session['file_name'])

		session['low'] = form.lower_bound_time.data
		session['up'] = form.upper_bound_time.data
		session['n_comps'] = int(form.n_components.data)
		session['spect_choice'] = int(form.spec_opts.data)
		session['dcomp_type'] = int(form.d_opts.data)
		samples, rate = load_sample(path='uploads/'+str(session['file_name']), upper_bound = session['up'], lower_bound = session['low'])

		
		fourier_transform = get_fft(samples,rate) 
		peaks = get_peak_frequencies(fourier_transform['y'])
		freqs = [p[0] for p in peaks]
		notes = [p[1] for p in peaks]
		chord = get_chord(notes)

		float_freqs = [float(f) for f in freqs]
		f_min = int(np.min(float_freqs)) # get this to add a lower bound to the spectrogram
			
		
		plt.figure(figsize=(14,12))
		
		# waveform amplitude
		plt.subplot(3,1,1,title='Waveform',xlabel='Time (s)',ylabel='Amplitude')
		plt.plot(np.arange(0,len(samples))/rate, samples) 

		# fourier transform
		plt.subplot(3,1,2, title='Fourier Transform', xlim=(20,20000), xlabel='Hz', ylabel='Relative Amplitude')
		plt.semilogx(fourier_transform['x'], fourier_transform['y'])

		if session['spect_choice'] == 1:
			plt.subplot(3,1,3, title='Spectrogram (Log Scale)')
			ld.specshow(amplitude_to_db(np.abs(lc.stft(samples)),ref=np.max), y_axis='log', x_axis='time', cmap=cmap) 

		elif session['spect_choice'] == 2:
			plt.subplot(3,1,3, title='Spectrogram (Linear Scale)') # placeholder, figure out how to do mel-scale/if it's useful/any other sort of plot
			ld.specshow(amplitude_to_db(np.abs(lc.stft(samples)),ref=np.max), y_axis='linear', x_axis='time', cmap=cmap)

		elif session['spect_choice'] == 3:
			plt.subplot(3,1,3, title='Constant Q Spectrogram')
			ld.specshow(amplitude_to_db(np.abs(lc.cqt(samples)),ref=np.max), y_axis='log', x_axis='time',  cmap=cmap)

		elif session['spect_choice'] == 4:
			plt.subplot(3,1,3, title='Chromagram')
			ld.specshow(amplitude_to_db(np.abs(chroma_stft(samples)),ref=np.max), y_axis='chroma', x_axis='time', cmap=cmap)

		zcr = zero_crossing_rate(y=samples+0.01)[0]
		sc = spectral_centroid(y=samples+0.01,sr=rate)[0]

		plt.tight_layout()
		plot_file = 'static/plot_' + curr_time + '.png'
		plt.savefig(plot_file)

		session['freqs'] = freqs
		session['notes'] = notes
		session['chord'] = chord
		session['plots'] = plot_file

		return redirect(url_for('basic_plots'))

	return render_template('upload.html', form=form)


@app.route('/basic_plots',methods=['GET','POST'])
def basic_plots():
	return render_template('basic_plots.html')


@app.route('/features',methods=['GET','POST'])
def features():
	# display a page with the plots of the zcr and spectral centroid (possibly other spectral features) variations here
	f = extract_features(signal = session['samples'])
	return None


@app.route('/harm_perc',methods=['GET','POST'])
def harm_perc():
	curr_time = str(time.time())[:10] # need to reload samples because it's too big to be saved as a session cookie
	samples, rate = load_sample(path='uploads/'+str(session['file_name']), upper_bound = session['up'], lower_bound = session['low']) 
	
	harmonic = librosa.effects.harmonic(samples, margin = 3.0, kernel_size=31, power=2.0)
	percussive = librosa.effects.percussive(samples, margin = 3.0, kernel_size=31, power=2.0)
	
	h_fourier_transform = get_fft(harmonic,rate)
	p_fourier_transform = get_fft(percussive,rate)

	plt.figure(figsize=(18,18))

	ax1 = plt.subplot2grid((4, 2), (0, 0))
	ax2 = plt.subplot2grid((4, 2), (0, 1))
	ax3 = plt.subplot2grid((4, 2), (1, 0))
	ax4 = plt.subplot2grid((4, 2), (1, 1))
	ax5 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
	ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)

	ax1.plot(np.arange(0,len(harmonic))/rate, harmonic) 
	ax1.set_title('Harmonic Components Waveform')
	ax1.set_xlabel('Time (s)')
	ax1.set_ylabel('Amplitude')
	
	ax2.semilogx(h_fourier_transform['x'], h_fourier_transform['y'])
	ax2.set_title('Harmonic Components Fourier Transform')
	ax2.set_xlim([20,20000])
	ax2.set_xlabel('Hz')
	ax2.set_ylabel('Relative Amplitude')
	
	ax3.plot(np.arange(0,len(percussive))/rate, percussive) 
	ax3.set_title('Percussive Components Waveform')
	ax3.set_xlabel('Time (s)')
	ax3.set_ylabel('Amplitude')

	ax4.semilogx(p_fourier_transform['x'], p_fourier_transform['y'])
	ax4.set_title('Percussive Components Fourier Transform')
	ax4.set_xlim([20,20000])
	ax4.set_xlabel('Hz')
	ax4.set_ylabel('Relative Amplitude')
	
	ld.specshow(amplitude_to_db(np.abs(lc.stft(harmonic)),ref=np.max), y_axis='log', x_axis='time', cmap=cmap, ax=ax5)
	ax5.set_title('Harmonic Components Spectogram')
	ax5.set_xlabel('Time (s)')
	ax5.set_ylabel('Hz')
	
	ld.specshow(amplitude_to_db(np.abs(lc.stft(percussive)),ref=np.max), y_axis='log', x_axis='time', cmap=cmap, ax=ax6)
	ax6.set_title('Percussive Components Spectogram')
	ax6.set_xlabel('Time (s)')
	ax6.set_ylabel('Hz')
	
	plt.tight_layout()

	hp_plot_file = 'static/hp_plot_' + curr_time + '.png'
	plt.savefig(hp_plot_file)
	session['hp_plots'] = hp_plot_file	

	h_file = 'static/h_' + curr_time + '.wav'
	p_file = 'static/p_' + curr_time + '.wav'
	write_wav(path=h_file, y=harmonic, sr=rate)
	write_wav(path=p_file, y=percussive, sr=rate)
	session['harmonic'] = h_file
	session['percussive'] = p_file
	return render_template('hp_plots.html')


@app.route('/components',methods=['GET','POST'])
def components():
	curr_time = str(time.time())[:10] 
	# need to reload samples because it's too big to be saved as a session cookie
	samples, rate = load_sample(path='uploads/'+str(session['file_name']), upper_bound = session['up'], lower_bound = session['low']) 
	stft = lc.stft(samples)
	X, phase = lc.magphase(stft)
	nc = session['n_comps']
	dco = session['dcomp_type']

	if dco == 1:
		T = sklearn.decomposition.NMF(n_components=nc,solver='cd', beta_loss='frobenius') 
	elif dco == 2:
		T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=nc,fit_algorithm='lars',transform_algorithm='omp')
	elif dco == 3:
		T = sklearn.decomposition.PCA(n_components=nc,svd_solver='auto') 
	elif dco == 4:
		T = sklearn.decomposition.FastICA(n_components=nc,algorithm='parallel',whiten=True,fun='logcosh')

	W, H = librosa.decompose.decompose(X,n_components=nc,sort=True,transformer=T) # pass in decomposition option T here
	components = {} # stft matrices
	component_samples = {} # individual sample .wav files
	session['component_files'] = [] # names of files referring to
	
	for i in range(nc):
		components[i] = np.dot(np.reshape(W.T[i],(W.shape[0],1)), np.reshape(H[i],(1,H.shape[1]))) * phase # "outer" dot product of each component
		component_samples[i] = lc.istft(components[i]) # get audio sample for each component
		session['component_files'].append('static/comp_' + str(i) + '_' + curr_time + '.wav')
		write_wav(path=session['component_files'][i],y=component_samples[i],sr=rate)

	plt.figure(figsize=(14,12)) 
	plt.suptitle(t=dco_to_title[dco],y=1.0)

	for i in range(nc):
		plt.subplot(nc, 2, (2*i)+1, title='Component ' + str(i) + ' Frequencies', xlabel='Hz')
		harmonic = W.T[i]/np.max(W.T[i]) # normalize
		plt.semilogx(np.arange(0,len(harmonic)),harmonic)
		plt.subplot(nc, 2, (2*i)+2, title='Component ' + str(i) + ' Temporal Activations', xlabel='Time (s)')
		temporal = H[i]/np.max(H[i]) # normalize
		plt.plot(np.arange(0,len(temporal))/rate,temporal)


	plt.tight_layout()
	cplot_file = 'static/component_plot_' + curr_time + '.png'
	plt.savefig(cplot_file)
	session['component_plots'] = cplot_file	
	return render_template('component_plots.html')


if __name__=='__main__':
	app.run(debug=True)
