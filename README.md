# BandwidthExtension

[Bandwidth extension](https://en.wikipedia.org/wiki/Bandwidth_extension) of the signal is defined as the deliberate process of expanding the frequency range (bandwidth) of a signal in which it contains an appreciable and useful content and/or the frequency range in which its effects are such.


## Abstract
  >Advancements in the field of deep learning have led to some remarkable discoveries in
 the field of audio processing. A plethora of resources makes it possible to implement them
 and supplies the opportunity to present a better version of previously developed algorithms and methods. In this thesis, we present an implementation of Bandwidth Extension for audio using generative models with the help of an end-to-end based deep learning model using the Pytorch deep learning library.

>With the use of deep learning-based study, we have studied multiple neural network models, with variations in the input data to the model, for a better understanding of the underlying structure in the audio data and how the structure can be exploited for best results. In addition to that, models were trained against different loss functions. Loss functions play a huge role in supplying better results. One of the loss functions we considered was based on the perception of sound by the human ear, known as Weighted Mean Square Error(WMSE) because generic loss functions such as Mean Squared Error(MSE) are insufficient for audio synthesis. Hence including perception-based error function proves to be better than MSE and provides a better reconstruction of high-frequency components than MSE-based reconstruction. Another error function that generated better results was Log Spectral Distance(LSD). It was compared with the other loss functions across Polyphase and Non-Polyphase based RNN-Autoencoder.

> Models considered for this evaluation are SampleRNN and an RNN Autoencoder, which utilizes Resnet. Multiple experiments for bandwidth extension were performed for various sampling rates. Sampling rates considered for these experiments are 4 kHz to 8kHz, 8 kHz to 16 kHz, and 16kHz to 32 kHz. Predicted higher frequency components from low-frequency components were confirmed by looking at reconstructed spectrograms. In addition, a Mushra Test was performed to evaluate the quality of reconstructed audio samples for 8kHz as input and 16kHz as a target for the experiments mentioned above. As per Mushra test and spectrograms, Non-Polyphase based RNN-Autoencoder using WMSE as an error function proved to be closest to target audio data. 

## Block Diagram
![Simple Block Diagram](imgs/base.PNG "Simple Block Diagram")

## Models Examined 
1. Modified Unconditional SampleRNN
2. RNN-Autoencoder
    1. Non-Polyphase based RNN-Autoencoder
    2. Polyphase based RNN-Autoencoder
## Errors Examined
1. Mean Square Error(MSE)
2. Weighted Mean Square Error(WMSE)
3. Logarthmic Spectral Distance(LSD)


![ORP](imgs/Orig_Ac_corrr.png "Original vs PAC Reconstructed and White Noise Corrupted Audio")


|      | Original vs Reconstructed | Original vs White Noise Corrupted |
|------|---------------------------|-----------------------------------|
| MSE  |          0.01886          |              0.01886              |
| WMSE |          1.46956          |              35.06760             |
| LSD  |          0.89329          |              2.47500              |

### Observation
1. MSE for both of the cases is same and is not able to diffrentiate between them, on the other hand, WMSE and LSD is able to diffrentiate between the white noise corrupted audio and Reconstructed audio.


2. The image shows audio samples generated via Perceptual Audio Coder and and White Noise corrupted by same SNR as of input and audio coder reconstructed.

## Modified Unconditional SampleRNN
![Samplernn](imgs/rsz_modifiedsamplernn.png "Modified SampleRNN for BandwidthExtension") 
#### Details

## RNN-Autoencoder

### Non-Polyphase based RNN-Autoencoder 
![Non-Polyphase](imgs/rsz_transposedconvolution_autorncoder_rnn_np_01.png "Non-Polyphase")  
#### Details
> Diese rasch um jahre ja lagen es du sitte. Winter freute weg wei konnen burger vielen. War hemdarmel liebevoll verharrte das sorglosen. So ab sa horst en diese genug. Betrubte geworden wei wie indessen kindbett funkelte. Wu stillen uberall gewerbe fenster an mi. Gehe lief eck sohn nun lich also mann. Hab entgegen getraumt zinnerne spielend als burschen.

> So eleganz spiegel heimweh es. Horchend gefallen achtzehn schlafer he vergrast je. Gerufen gefreut verband es stopfen so. La licht so mi ruhen weste je. Pa nachdem dichten anblick ku so lockere obenhin. Sudwesten gestrigen schnupfen tat ist besserung. Tur nest vor hast hol fast lass kerl weil. Sa kinder se es sofort an schade.


### Polyphase based RNN-Autoencoder
![Polyphase](imgs/rsz_transposedconvolution_autorncoder_rnn_p_02.png "Polyphase")
#### Details
> Diese rasch um jahre ja lagen es du sitte. Winter freute weg wei konnen burger vielen. War hemdarmel liebevoll verharrte das sorglosen. So ab sa horst en diese genug. Betrubte geworden wei wie indessen kindbett funkelte. Wu stillen uberall gewerbe fenster an mi. Gehe lief eck sohn nun lich also mann. Hab entgegen getraumt zinnerne spielend als burschen.

> So eleganz spiegel heimweh es. Horchend gefallen achtzehn schlafer he vergrast je. Gerufen gefreut verband es stopfen so. La licht so mi ruhen weste je. Pa nachdem dichten anblick ku so lockere obenhin. Sudwesten gestrigen schnupfen tat ist besserung. Tur nest vor hast hol fast lass kerl weil. Sa kinder se es sofort an schade.

## Results



To be updated soon