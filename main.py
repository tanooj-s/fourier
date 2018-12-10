
from flask import url_for, redirect, render_template, Flask, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug import secure_filename

import time
import numpy as np
import pandas as pd
import librosa.core as lc
from librosa import amplitude_to_db
import librosa.display
import librosa.effects
import librosa.decompose

import matplotlib.pyplot as plt

from scipy import fftpack

class UploadForm(FlaskForm):
	user_file = FileField()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AHs0JleAhvzcolWG'
app.config['UPLOADED_FILES_DEST'] = 'uploads/'

@app.route('/',methods=['GET','POST'])
def upload():
	form = UploadForm()
	if form.validate_on_submit():
		filename = secure_filename(form.user_file.data.filename)
		form.user_file.data.save('uploads/' + filename)
		# now the file has been uploaded by the user, we need to read it in from the uploads folder using librosa
		curr_time = str(time.time())[:10] # get epoch time without decimals to save to plt image later
		samples, rate = lc.load(path='uploads/'+str(filename),duration=15,mono=True)
		samples, _ = librosa.effects.trim(samples,top_db=20)
		fourier_transform = get_fft(samples,rate)
		constant_q = librosa.amplitude_to_db(np.abs(lc.cqt(samples)),ref=np.max)
		# get harmonic and percussive components using built in librosa 
		h,p = librosa.amplitude_to_db(librosa.decompose.hpss(np.abs(lc.stft(samples))),ref=np.max) 
		
		fig = plt.figure(figsize=(15,12))	
		axis1 = fig.add_subplot(4,1,1)
		axis1.set_title('Constant Q Transform')
		axis1.set_xlabel('Time (s)')
		axis1.set_ylabel('Notes - 12 per octave')
		axis1.pcolormesh(constant_q)
		axis1.set_yticklabels(np.arange(0,constant_q.shape[1],12))


		# in addition to a plot what this needs is a widget that allows you to pay the decomposed audio files 
		# (and possibly a download link for them)
		axis2 = fig.add_subplot(4,1,2)
		axis2.set_title('Harmonic Components')
		axis2.pcolormesh(h)
		# need to scale the y axis to log scale here
		axis3 = fig.add_subplot(4,1,3)
		axis3.set_title('Percussive Components')
		axis3.pcolormesh(p)

		axis4 = fig.add_subplot(4,1,4)
		axis4.set_ylabel('Relative Amplitude')
		axis4.set_xlabel('Frequency (Hz)')
		axis4.set_xlim(20,20000) # human range
		axis4.set_xticklabels(np.arange(100,20000,100))
		axis4.semilogx(fourier_transform['x'],fourier_transform['y'])

		fig.tight_layout()
		plot_path = 'static/plot_' + curr_time + '.png'
		fig.savefig(plot_path)
		#each plot should be its own image, separate objects for spectrograms and frequency-space plots

		session['path_to_plot'] = plot_path		
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
	yfft = np.abs(yfft[:nyquist])/(max_amp) # not sure why we don't need to multiply by 2/len(samples) to scale properly here
	return {'x': xfft, 'y': yfft}


if __name__=='__main__':
	app.run(debug=True)