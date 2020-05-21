This (currently) super stripped down web app allows a user to upload an audio file and returns a plot of the Fourier transform of the specified duration of the sample as well as a spectrogram of the percussive and harmonic components of the file and a chromagram to display the "density" of notes at each frame. The file is bounced down to mono and read in at a sample rate of 22050 Hz.

Users now also have the option to download audio files of the separated harmonic and percussive components.
Examples output files can be found in the static folder. Basic chord identification has been added, although not for separate frames yet (i.e. it assumes the entire uploaded audio file is a single chord).

Features to add: useful information retrieval features, possibly chord/key recognition or an instrument source separator ( - pipe dream!) 

