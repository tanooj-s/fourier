
from flask import url_for, redirect, render_template, Flask, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from werkzeug import secure_filename

import time
import numpy as np
import pandas as pd
import librosa.core as lc
from librosa import amplitude_to_db
from matplotlib.figure import Figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import librosa.display
from scipy import fftpack

class UploadForm(FlaskForm):
	user_file = FileField()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AHs0JleAhvzcolWG'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads/'

@app.route('/',methods=['GET','POST'])
def upload():
	form = UploadForm()
	if form.validate_on_submit():
		filename = secure_filename(form.user_file.data.filename)
		form.user_file.data.save('uploads/' + filename)
		# now the file has been uploaded by the user, we need to read it in from the uploads folder using librosa
		curr_time = str(time.time())[:10] # get epoch time without decimals to save to plt image later
		samples, rate = lc.load(path='uploads/'+str(filename),duration=15,mono=True)

		nyquist = int(rate/2)
		yfft = fftpack.fft(x=samples,n=rate)
		max_amp = np.max(np.abs(yfft))
		xfft = np.linspace(0,nyquist,nyquist)
		yfft = np.abs(yfft[:nyquist])/(max_amp) # not sure why we don't need to multiply by 2/len(samples) to scale properly here
		
		fig = plt.figure(figsize=(15,6))
		axis = fig.add_subplot(1,1,1)
		axis.set_xlabel('Frequency (Hz)')
		axis.set_ylabel('Relative Amplitude')
		axis.set_xlim(20,20000) # human range
		axis.set_xticklabels(np.arange(100,20000,100))
		axis.semilogx(xfft,yfft)
		plot_path = 'static/img/plot_' + curr_time + '.png'
		fig.savefig(plot_path)

		session['path_to_plot'] = plot_path		
		return redirect(url_for('fast_fourier'))
	return render_template('upload.html', form=form)


@app.route('/fft',methods=['GET','POST'])
def fast_fourier():
	return render_template('fft_plot.html')


if __name__=='__main__':
	app.run(debug=True)