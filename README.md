# Project Title: Exploring Generative AI (Variational AutoEncoders) for Audio Generation Task

Speech is a unique trait inherent to humans, allowing us to convey thoughts and ideas through vocal sounds. It serves as the most natural method of communication among humans. And even in the area of human-computer interaction, attempts have been made to integrate speech. However, the main issue within this implementation is that all the verbal replies we get from computers, intelligent systems, or artificial intelligence systems still need to be pre-recorded, and the computers themselves do not generate speech data. Hence, there is a need to generate speech data that works without the restrictions possessed by the prerecorded audio data.

This project explores the application of Variational Autoencoders (VAEs) in the domain of audio generation, specifically focusing on generating spoken digits. The study employs VAE architectures with Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) layers for encoding and decoding audio data represented in Short-Time Fourier Transform (STFT) features. A detailed analysis of the trained VAE models is conducted, investigating various training epochs, regularization techniques, and model architectures. The performance of each model variant is evaluated based on training time, loss metrics, and latent space representations. The outcomes provide insights into the efficacy of VAEs for generating high-fidelity audio samples and their ability to learn distinctive representations for spoken digits.

## Dataset
A simple audio dataset consisting of recordings of spoken digits in wav files  sampled at 48kHz. The recordings are processed (filtered and trimmed) so that the filtered samples have minimal silence at the beginning and end.
* **Format :** ".wav"
* **Sampling Rate:** 48000 Hz (resampled to 22050 Hz)
* **# of Samples :** 3000
* **# of Speakers :** 60

[Link to Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)


## Outputs
Actual Audio:  
Generated Audio:






## How to use the code?

create python environment
```bash
conda create --name myenv
```

install python libraries
```bash
pip3 install -r codes/requirements.txt
```

train model
```bash
python3 codes/train.py
```
generate audio (give input as a string of digits that you want to generate)
```bash
python3 codes/inference.py "1"
```

### Tech Stack
• Keras
• Tensorflow
• Pickle
• Glob
• Librosa
• Tqdm
• Pandas
• Numpy
• SciPy
• Math
• Sklearn
• Matplotlib
• IPython

### Contributors: 
* Rabin Nepal [(Bot-Ro-Bot)](https://github.com/Bot-Ro-Bot)
* Sravan Kumar Dumpeti
