# MIC
Music Generation with Text and Audio Conditioning
This project focuses on building a deep learning model to generate musical sequences. The generation process is influenced by two conditioning inputs: textual descriptions (like lyrics or prompts) and features extracted from an audio file. The core of the system is a Bidirectional GRU model implemented in PyTorch, designed to learn the complex relationships between these conditioning inputs and musical features (specifically, MFCCs). The generated MFCC sequences are then transformed back into audible audio. Additionally, an experimental component attempts to convert these features into a symbolic music representation (MIDI) using the music21 library.

Features
Text Conditioning: This feature allows users to guide the music generation process using text. You can provide lyrical content or a descriptive prompt (e.g., "upbeat electronic dance track," "calm piano melody"). A language model (currently using DistilGPT-2 via the Hugging Face transformers library) is used to either generate lyrics based on a prompt or process provided lyrics. A placeholder mechanism is used to obtain a fixed-size numerical embedding from this text, which serves as a conditioning input for the music generation model.

Audio Conditioning: The generated music can also be conditioned on a user-provided audio file. This means the model will attempt to generate music that is musically related to the input audio. The audio conditioning is achieved by extracting a sequence of features from the input audio file. These features, specifically Mel-Frequency Cepstral Coefficients (MFCCs) and Chroma features, capture important timbral and harmonic characteristics of the audio.

Iterative Generation: To overcome the limitation of generating fixed-length sequences, the project employs an iterative generation process. The trained PyTorch model predicts a short segment of musical features based on the current sequence and conditioning inputs. This predicted segment is then used as part of the input for the next prediction step, allowing the generation of longer musical sequences step-by-step. This creates a continuous flow of generated music up to a specified target duration.

Feature Extraction: Audio feature extraction is a crucial preprocessing step. The project uses the Librosa library to extract two key types of features:

MFCCs (Mel-Frequency Cepstral Coefficients): 20 coefficients are extracted per audio frame. These capture the spectral envelope of the sound, which is important for representing timbre.
Chroma Features: 12 chroma bins are extracted per audio frame. These represent the distribution of energy across the 12 pitch classes of the chromatic scale, providing information about the harmonic content. The extracted features (concatenated MFCCs and Chroma) are then padded or truncated to a fixed max_padding length (currently 174 time steps) to create consistent input sequences for the model. Only MFCCs are used as the target output for the model to predict.
Audio Synthesis: The generated musical features (which are sequences of MFCCs) are converted back into audible audio using the inverse MFCC transform provided by Librosa. This process reconstructs an audio time series from the spectral features. The generated audio is then saved to files in both WAV format (using Soundfile) and MP3 format (using Pydub for conversion).

Symbolic Music Conversion (Experimental and In Development): This is an experimental feature that attempts to bridge the gap between audio features and symbolic music notation. The project includes a basic function to convert the generated MFCCs into a music21 stream. Currently, this involves a simple mapping of the first MFCC coefficient to a MIDI pitch and using a fixed duration based on the hop length. It's important to note that directly converting complex audio features like MFCCs to musically meaningful notes, rhythms, and harmony is a challenging task. This basic implementation is a starting point and requires significant further development and potentially more sophisticated techniques or a dedicated model trained specifically for feature-to-symbolic conversion to produce musically coherent results. The generated symbolic music can be saved as a MIDI file using music21.

Technologies Used
Python: The primary programming language.
PyTorch: Used for building and training the deep learning model (Bidirectional GRU).
TensorFlow/Keras: Used in earlier examples for model definition, but the main conditional generation model is implemented in PyTorch.
NumPy: Essential for numerical operations and handling feature arrays.
Librosa: A powerful library for audio analysis, feature extraction (MFCC, Chroma), and audio synthesis (inverse MFCC).
Soundfile: Used for reading and writing audio files (specifically WAV).
Pydub: Used for audio manipulation, including converting the generated WAV file to MP3 format.
Transformers (Hugging Face): Used for the text generation pipeline (DistilGPT-2) to generate lyrics from prompts.
scikit-learn: Used for splitting the dataset into training and validation sets.
KaggleHub: Used to download the Musical Instruments Sound Dataset.
music21: A toolkit for creating, manipulating, and analyzing musical notation. Used here experimentally for converting generated features to a symbolic representation and saving as MIDI.
IPython: Provides enhanced interactive features for the notebook environment, including displaying audio players and file links.
Dataset
The project utilizes the Musical Instruments Sound Dataset available on Kaggle. This dataset contains a variety of audio recordings of different musical instruments, which are used to train the model to learn musical patterns and characteristics.
#UPDATE TO THE MIC WILL UPDATE SOON WILL ADD SOME MORE DATA AND FINE TUNE IT AND MAKE IT COMPLETE IT PROPER LLM WITH SLM ABILITY .
