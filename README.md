# Project Title : Exploring Generative AI (Variational AutoEncoders) for Audio Generation Task

Speech is a unique trait inherent to humans, allowing us to convey thoughts and ideas through vocal sounds. It serves as the most natural method of communication among humans. And even in the area of human-computer interaction, attempts have been made to integrate speech. However, the main issue within this implementation is that all the verbal replies we get from computers, intelligent systems, or artificial intelligence systems still need to be pre-recorded, and the computers themselves do not actually generate speech data. Hence, there is a need to generate speech data that works without the restrictions possessed by the prerecorded audio data.

This project will explore the usage of Generative models on audio data to shed more light on the potential usage of such models to generate speech, which is a more inherent form of communication to humans. This project's scope is to explore the use of generative AI models, more specifically the VAE (Variational AutoEncoders) model to see how well these models work on audio generation tasks. Through this exploration, the project aims to contribute to the broader understanding of variational autoencoders and their role in audio generation.


## Dataset
A simple audio dataset consisting of recordings of spoken digits in wav files  sampled at 48kHz. The recordings are processed (filtered and trimmed) so that the filtered samples have minimal silence at the beginning and end.
* **Format :** ".wav"
* **Sampling Rate:** 48000 Hz (resmapled to 22050 Hz)
* **# of Samples :** 3000
* **# of Speakers :** 60

[Link to Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)


## Objective
* Explore the application of generative models, specifically Variational AutoEncoders (VAE), in audio data.
* Focus on speech generation and spontaneity of human speech.
* Contribute insights to the understanding of variational autoencoders and their role in audio generation task


## Methodology



## Results

### MLP-VAE Model


### CNN-VAE Model

## How to use the code?

create python environment
'''

'''

install python libraries
'''
pip3 install -r codes/requirements.txt
'''

train model
'''
python3 codes/train.py
'''
generate audio (give input as string of digit that you want to generate)
'''
python3 codes/inference.py "1"
'''




### Tech Stack
* SciPy 
* Pandas
* Librosa
* Keras
* Sklearn
* TensorFlow



### References
* [Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
* https://www.deeplearningbook.org/
* Valerio-The Sound of AI
