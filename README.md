This (currently) super stripped down web app allows a user to upload an audio file and returns a plot of the Fourier transform of the first 15 seconds of the sample and a spectrogram of the constant-Q transform (this works better than the Short Time Fourier Transform for musical applications). The file is bounced down to mono and read in at a sample rate of 22050 Hz, although I'll add more options for the user once I have an actual input form on the homepage. I will also add the option to try out different kinds of plots (at the very least a spectrogram and a constant-Q transform). 

![Example output](https://github.com/tanooj-s/fourier/tree/master/static/plot_1544391484.png)

Medium term to long term I hope to add some useful information retrieval features, possibly chord/key recognition or an instrument source separator. 

