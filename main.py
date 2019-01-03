
from flask import url_for, redirect, render_template, Flask, session
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField
from flask_wtf.file import FileField
from werkzeug import secure_filename

import time
import numpy as np
import librosa.core as lc
from librosa import amplitude_to_db
import librosa.display
import librosa.effects
import librosa.decompose
from librosa.feature import chroma_stft
from librosa.output import write_wav

import matplotlib.pyplot as plt

from scipy import fftpack

class UploadForm(FlaskForm):
	user_file = FileField()
	lower_bound_time = FloatField()
	upper_bound_time = FloatField()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AHs0JleAhvzcolWG'
app.config['UPLOADED_FILES_DEST'] = 'uploads/'

@app.route('/',methods=['GET','POST'])
def upload():
	form = UploadForm()

	if form.validate_on_submit():
		# save user's uploaded file
		file_name = secure_filename(form.user_file.data.filename)
		form.user_file.data.save('uploads/' + file_name)
		# get user inputted upper and lower bounds
		low = form.lower_bound_time.data
		up = form.upper_bound_time.data
		
		curr_time = str(time.time())[:10] # get epoch time without decimals to save to files later
		samples, rate = lc.load(path='uploads/'+str(file_name),mono=True,duration=up)

		samples, _ = librosa.effects.trim(samples[int(rate*low):int(rate*up)],top_db=20)

		harmonic = librosa.effects.harmonic(samples, margin = 2.0)
		percussive = librosa.effects.percussive(samples, margin = 2.0) 
		fourier_transform = get_fft(harmonic,rate) # only get the fourier transform of the harmonic

		# get peak frequencies from 
		peaks = get_peak_frequencies(fourier_transform['y'])
		
		cmap = plt.get_cmap('Blues')
		plt.figure(figsize=(12,15))

		# plot waveform amplitude
		plt.subplot(4,1,1, title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
		plt.plot(np.arange(0,len(samples))/rate, samples)

		# plot fourier transform
		plt.subplot(4,1,2, title='Fourier Transform', xlim=(20,20000), xlabel='Hz', ylabel='Relative Amplitude')
		plt.semilogx(fourier_transform['x'], fourier_transform['y'])
		# try doing this only for harmonic components instead

		# plot spectrogram of harmonic components
		plt.subplot(4,1,3, title='Spectrogram')
		librosa.display.specshow(amplitude_to_db(np.abs(lc.stft(harmonic)),ref=np.max), y_axis='log', x_axis='time', cmap=cmap)

		# plot percussive components 
#		plt.subplot(5,1,4, title='Percussive Components')
#		librosa.display.specshow(amplitude_to_db(np.abs(lc.stft(percussive)),ref=np.max), y_axis='log', x_axis='time', cmap=cmap)

		# plot chromagram of harmonic components (try different librosa variants)
		chromagram = chroma_stft(harmonic, rate, hop_length=512)
		plt.subplot(4,1,4, title='Chromagram')
		librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', cmap=cmap)

		plt.tight_layout()
		plot_file = 'static/plot_' + curr_time + '.png'
		plt.savefig(plot_file)

		# save harmonic and percussive breakdown for user to download
		h_file = 'static/h_' + curr_time + '.wav'
		p_file = 'static/p_' + curr_time + '.wav'
		write_wav(path=h_file, y=harmonic, sr=rate)
		write_wav(path=p_file, y=percussive, sr=rate)


		session['peaks'] = peaks
		session['plots'] = plot_file
		session['harmonic'] = h_file
		session['percussive'] = p_file

		return redirect(url_for('results'))

	return render_template('upload.html', form=form)


@app.route('/plots',methods=['GET','POST'])
def results():
	return render_template('results.html')


def get_fft(samples,rate):
	nyquist = int(rate/2)
	yfft = fftpack.fft(x=samples,n=rate)
	max_amp = np.max(np.abs(yfft))
	xfft = np.linspace(0,nyquist,nyquist)
	yfft = np.abs(yfft[:nyquist])/(max_amp) 
	return {'x': xfft, 'y': yfft}


def get_peak_frequencies(lst):
	# for this use case lst is assumed to be fft['y'] i.e. the relative amplitudes of the fourier tranform
	indices = sorted(np.argpartition(lst,-10)[-10:]) # get indices of 15 max values in the list (these essentially correspond to fft['x'])
	peak_freqs = []
	for i in indices:
		if lst[i] > 0.3:
			peak_freqs.append(i)
	peak_freqs = purge(peak_freqs) # only retain mean of clusters
	peak_freqs = [str(p) for p in peak_freqs]
	return peak_freqs

def purge(lst):
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

if __name__=='__main__':
	app.run(debug=True)