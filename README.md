# BandwidthExtension

[Bandwidth extension](https://en.wikipedia.org/wiki/Bandwidth_extension) of the signal is defined as the deliberate process of expanding the frequency range (bandwidth) of a signal in which it contains an appreciable and useful content and/or the frequency range in which its effects are such.

##### Google Colab file for inference using a pretrained model for Non-Polyphase based Model
> Model was trained for 16kHz input audio samples and 32kHz target audio samples using WMSE as a loss function
 
[![Inference example in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wt0t2uKdKv2qVse6ejxGFtdvLkGhxFc6#scrollTo=qQOKDlzGhV01&uniqifier=1)
##### A small website for input, target and generated audio samples.
[BandwidthExtension](https://mukul74.github.io/BandwidthExtensionAudioSamples/)
> Contains audio samples for
> 1. Female Voice + Instruments
> 2. Male Voice + Instruments
> 3. Genre Disco 

## Abstract
  >Advancements in the field of deep learning have led to some remarkable discoveries in
 the field of audio processing. A plethora of resources makes it possible to implement them
 and supplies the opportunity to present a better version of previously developed algorithms and methods. In this thesis, we present an implementation of Bandwidth Extension for audio using generative models with the help of an end-to-end based deep learning model using the Pytorch deep learning library.

>With the use of deep learning-based study, we have studied multiple neural network models, with variations in the input data to the model, for a better understanding of the underlying structure in the audio data and how the structure can be exploited for best results. In addition to that, models were trained against different loss functions. Loss functions play a huge role in supplying better results. One of the loss functions we considered was based on the perception of sound by the human ear, known as Weighted Mean Square Error(WMSE) because generic loss functions such as Mean Squared Error(MSE) are insufficient for audio synthesis. Hence including perception-based error function proves to be better than MSE and provides a better reconstruction of high-frequency components than MSE-based reconstruction. Another error function that generated better results was Log Spectral Distance(LSD). It was compared with the other loss functions across Polyphase and Non-Polyphase based RNN-Autoencoder.

> Models considered for this evaluation are SampleRNN and an RNN Autoencoder, which utilizes Resnet. Multiple experiments for bandwidth extension were performed for various sampling rates. Sampling rates considered for these experiments are 4 kHz to 8kHz, 8 kHz to 16 kHz, and 16kHz to 32 kHz. Predicted higher frequency components from low-frequency components were confirmed by looking at reconstructed spectrograms. In addition, a Mushra Test was performed to evaluate the quality of reconstructed audio samples for 8kHz as input and 16kHz as a target for the experiments mentioned above. As per Mushra test and spectrograms, Non-Polyphase based RNN-Autoencoder using WMSE as an error function proved to be closest to target audio data. 

## Block Diagram
![Simple Block Diagram](imgs/base.PNG "Simple Block Diagram")

## Models Examined 
0. Base Model  [Paper](https://kuleshov.github.io/audio-super-res/)
1. Modified Unconditional SampleRNN [Paper](https://arxiv.org/abs/1612.07837)
2. RNN-Autoencoder
    1. Non-Polyphase based RNN-Autoencoder
    2. Polyphase based RNN-Autoencoder
## Errors Examined
1. Mean Square Error(MSE)
2. Weighted Mean Square Error(WMSE)
3. Logarthmic Spectral Distance(LSD)



![ORP](imgs/Orig_Ac_corrr.png "Original vs PAC Reconstructed and White Noise Corrupted Audio")

##### Google Colab file for error comparision
[![Inference example in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AUALZF3DueLh9sQYaYN9Im3sIx9ej85R#scrollTo=3SREr21Vk93s)

|      | Original vs Reconstructed | Original vs White Noise Corrupted |
|------|---------------------------|-----------------------------------|
| MSE  |          0.01886          |              0.01886              |
| WMSE |          1.46956          |              35.06760             |
| LSD  |          0.89329          |              2.47500              |

### Observation
1. MSE for both of the cases is same and is not able to diffrentiate between them, on the other hand, WMSE and LSD is able to diffrentiate between the white noise corrupted audio and Reconstructed audio.
2. The image shows audio samples generated via Perceptual Audio Coder and and White Noise corrupted by same SNR as of input and audio coder reconstructed.
3. In the image it is visible there is quite a significant difference in the audio waveforms, but MSE is not able to differentiate between them.
4. MSE is of no use when it comes to perceptual difference in audio.
5. This small experiment drives the idea of Weighted Mean Squared Error and Log Spectral Difference.

## Modified Unconditional SampleRNN
![Samplernn](imgs/rsz_modifiedsamplernn.png "Modified SampleRNN for BandwidthExtension") 
#### Details
> A multi-scale RNN implementation helps to gather essential relationships among data, and we may train them on small samples for memory optimization. SampleRNN, unlike WaveNet, is a recurrent model. Recurrent cells, such as LSTMs or GRUs, can potentially spread meaningful information across indefinite time horizons. Discovering latent features over extended periods is difficult for LSTMs. It is primarily due to diminishing or exploding gradients. The basic concept driving SampleRNN is that audio shows prominent patterns at different timeframes and that these trends and features are constructed hierarchically. 
## RNN-Autoencoder

### Non-Polyphase based RNN-Autoencoder 
![Non-Polyphase](imgs/rsz_transposedconvolution_autorncoder_rnn_np_01.png "Non-Polyphase")  
#### Details
> After testing a simple one-layer RNN-Autoencoder, a more complex rnn autoencoder was implemented, with transposed convolution as an upsampling technique. We are keeping the encoder part of the rnn-autoencoder made of convolutional layers. The idea of Resnet was also extended in this model. The main idea behind providing residual connections is to help the model learn better for upsampling and use the learned features while downsampling. Since the input is raw audio data, a block length of 1024 samples was considered to have a 50\% overlap with the next block. These 1024 samples are provided to the model, and with the help of convolutional layers, we reduce the dimension of the input data. To add non-linearity, we use LeakyRelu in the encoder block, followed by a max pooling layer. Then we use an RNN layer as a bottleneck, and for the decoder block, we use Transposed Convolution. We use Relu and Dropouts in the decoder block, followed by the concatenation layer. This layer concatenates one of the outputs in the encoder block to the output of the decoder block. This concatenation is done channel-wise. At last, we flatten the output of the last convolution layer. We can visualize this implementation in the figure \ref{fig:nonpoly} below. Upscaling by a factor of 2 required the last convolution layer to predict 2048 values for 1024 values as input. As mentioned earlier, models were trained for WMSE and LSD separately. Appropriate changes were made for the loss function made to the training methods.

### Polyphase based RNN-Autoencoder
![Polyphase](imgs/rsz_transposedconvolution_autorncoder_rnn_p_02.png "Polyphase")
#### Details
> In the previous section, we saw a little complex version of the RNN-based autoencoder, which was trained for transposed convolution as an upsampling technique. In this version, we adopted the same idea, except for the input values to the network. In the previous section, audio samples of block length 1024 were used as an input to the network with 50% overlap. In this section, we downsampled the block of 1024 samples in a Polyphased. This block is divided into eight channels and 128 samples, then they are used as an input to the neural network model. The first convolutional layer is made in such a way that it can accommodate the incoming eight channels. This type of downsampling is termed polyphase. 1024 input samples are transformed as per the image shown below. The rest of the architecture RNN-Autoencoder is modified to attain the transformed input samples. Here also, we utilize LeakyRelu, maxpool, Relu, and Dropout in the same fashion as explained above. As mentioned earlier, models were trained for WMSE and LSD separately. Appropriate changes were made for the loss function made to the training methods.

#### Number of Parameters per Model
| Polyphase 	| Non-Polyphase 	|    Base   	| SampleRNN |
|-----------	|---------------	|-----------	|--------	|
| 0.09M     	| 2.3M          	| 72.06M    	| 78.02M 	|

#### Average inference time per Model in seconds
| Polyphase 	| Non-Polyphase 	| SampleRNN 	| Base     	|
|-----------	|---------------	|-----------	|----------	|
| 2.94   	    | 3.2        	    | 1.03   	    | 7.88  	|


## Experiments 
> ### **Experiments with different sampling rates were conducted, to evaluate our idea for a small dataset**

> > #### Experiments for Base model, SampleRNN based model, Non-Polyphase based model and Polyphase based RNN-Autoencoder for input sampled at 4kHz and target sampled at 8kHz



> > > #### 4kHz to 8kHz Average loss for Test Set having different genres 

> > >|            Models          	|     SpectralLoss Target_Input    	|     SpectralLoss Target_Predicted    	|               SpectralLoss   Target(IFR)_Predicted(IFR)         	|
> > >|:--------------------------:	|:--------------------------------:	|:------------------------------------:	|:---------------------------------------------------------------:	|
> > >|     4k_8k_Base             	|            2.838955498           	|              0.769846642             	|                            0.025563918                          	|
> > >|     4k_8k_WMSE_NON_POLY    	|            2.838955498           	|                 0.5532               	|                            0.013841194                          	|
> > >|     4k_8k_LSD_NON_POLY     	|            2.838955498           	|              0.336730437             	|                            0.018115259                          	|
> > >|     4k_8k_LSD_POLY         	|            2.838955498           	|              0.247910774             	|                            0.06388542                           	|
> > >|     4k_8k_MSE_NON_POLY     	|            2.838955498           	|              0.804133153             	|                           -0.075100321                          	|
> > >|     4k_8k_MSE_POLY         	|            2.838955498           	|              0.659037203             	|                           -0.078789327                          	|
> > >|     4k_8k_SampleRNN        	|            2.838955498           	|              -0.043308296            	|                           -0.056810266                          	|
> > >|     4k_8k_WMSE_POLY        	|            2.838955498           	|              0.326250678             	|                            0.028139032                          	|

> > > *Spectral Loss = mean(log10(abs(STFT_Y_trg) + 1e-7) - log10(abs(STFT_X_pred) + 1e-7))  
> > >*IFR(Input Frequency Range)

> > >| **Models**                      | **Blues **   | **Classic**  | **Country** | **Disco**   | **Hiphop**   | **Jazz**     | **Metal**   | **Pop**      | **Reggae**   |
> > >|:-------------------------------:|:------------:|:------------:|:-----------:|:-----------:|:------------:|:------------:|:-----------:|:------------:|:------------:|
> > >| **4k_8k_Base_10G**              | 0.70990538   | 0.634344068  | 0.818412703 | 0.838624215 | 0.816541922  | 0.602374846  | 0.96302852  | 0.79622038   | 0.761047393  |
> > >| **4k_8k_LSD_NonPolyphase_10G**  | 0.254802302  | 0.136714996  | 0.41278882  | 0.434504396 | 0.425998965  | 0.0978881    | 0.49370892  | 0.32702904   | 0.278526874  |
> > >| **4k_8k_LSD_Polyphase_10G**     | 0.149167863  | -0.004728154 | 0.304227284 | 0.318401845 | 0.292899534  | -0.042438272 | 0.413068879 | 0.202182852  | 0.169996922  |
> > >| **4k_8k_MSE_NonPolyphase_10G**  | 0.827463776  | 0.777132469  | 0.986265135 | 0.978288686 | 0.949135756  | 0.757457525  | 1.042413431 | 0.900481862  | 0.858593887  |
> > >| **4k_8k_MSE_Polyphase_10G**     | 0.471542123  | 0.35550147   | 0.61363126  | 0.618733776 | 0.611972332  | 0.329373513  | 0.715016741 | 0.5430022    | 0.495021099  |
> > >| **4k_8k_SampleRNN_10G**         | -0.169947441 | -0.476677227 | 0.03137644  | 0.049329544 | -0.012539472 | -0.407023662 | 0.130719438 | -0.093167413 | -0.149105346 |
> > >| **4k_8k_WMSE_NonPolyphase_10G** | 0.471709016  | 0.404018852  | 0.633616346 | 0.654599142 | 0.625983274  | 0.333949043  | 0.735064375 | 0.529633048  | 0.505101207  |
> > >| **4k_8k_WMSE_Polyphase_10G**    | 0.224478875  | 0.098066133  | 0.390202519 | 0.409518272 | 0.384841175  | 0.062868034  | 0.504171214 | 0.291417016  | 0.261211139  |




> #### Experiments for Non-Polyphase based RNN-Autoencoder, Polyphase based RNN-Autoencoder and Base model for higher sampling rates usch as 8kHz to 16kHz and 16kHz to 32kHz.

> ### 8kHz to 16kHz Average loss for Test Set

>|                                      	| MSE         	| WMSE     	| LSD         	|
>|--------------------------------------	|-------------	|----------	|-------------	|
>| Non-Polyphase_T_8k_16k_I_8k_16k_LSD  	| 0.0563799   	| 5.73E-07 	| 0.276729196 	|
>| Non-Polyphase_T_4k_8k_I_8k_16k_LSD   	| 0.051999187 	| 7.35E-07 	| 0.271316883 	|
>| Non-Polyphase_T_8k_16k_I_8k_16k_WMSE 	| 0.046481295 	| 4.03E-07 	| 0.261975849 	|
>| Non-Polyphase_T_4k_8k_I_8k_16k_WMSE  	| 0.079269871 	| 5.93E-07 	| 0.259256431 	|
>| Polyphase_T_8k_16k_I_8k_16k_LSD      	| 0.043998776 	| 7.21E-07 	| 0.277621126 	|
>| Polyphase_T_4k_8k_I_8k_16k_LSD       	| 0.026988551 	| 6.02E-07 	| 0.290659596 	|
>| Polyphase_T_8k_16k_I_8k_16k_WMSE     	| 0.035046323 	| 3.98E-07 	| 0.272616319 	|
>| Polyphase_T_4k_8k_I_8k_16k_WMSE      	| 0.005492212 	| 3.90E-07 	| 0.277659197 	|


 
> ### 8kHz to 16kHz Average loss for Test Set having different genres 
> *T_4k_8k_I_8k_16k means Trained for input samples at 4kHz and target of 8kHz and Inferred for 8kHz and 16kHz\
> *T_8k_16k_I_8k_16k means Trained for input samples at 8kHz and target of 16kHz and Inferred for 8kHz and 16kHz  

>|                  Models               |     SpectralLoss Target_Input  |     SpectralLoss Target_Predicted  |
>|-------------------------------------- |--------------------------------|------------------------------------|
>|T_4k_8k_I_8k_16k_LSD_NON_POLY      	|             2.73441422         |              0.270420129           |
>|T_4k_8k_I_8k_16k_LSD_POLY          	|             2.73441422         |              0.187405745           |
>|T_4k_8k_I_8k_16k_WMSE_NON_POLY     	|             2.73441422         |              0.465945017           |
>|T_4k_8k_I_8k_16k_WMSE_POLY         	|             2.73441422         |               0.25570653           |
>|T_8k_16k_I_8k_16k_Base             	|             2.73441422         |              0.603543603           |
>|T_8k_16k_I_8k_16k_LSD_NON_POLY     	|             2.73441422         |              0.225586262           |
>|T_8k_16k_I_8k_16k_LSD_POLY         	|             2.73441422         |              0.154223925           |
>|T_8k_16k_I_8k_16k_WMSE_NON_POLY    	|             2.73441422         |              0.405578698           |
>|T_8k_16k_I_8k_16k_WMSE_POLY        	|             2.73441422         |               0.17192617           |




> ###  16kHz to 32kHz Average loss for Test Set
> *T_4k_8k_I_16k_32k means Trained for input samples at 4kHz and target of 8kHz and Inferred for 16kHz and 32kHz\
> *T_16k_32k_I_16k_32k means Trained for input samples at 16kHz and target of 32kHz and Inferred for 16kHz and 32kHz
>|                                        	| MSE         	| WMSE     	| LSD      	|
>|----------------------------------------	|-------------	|----------	|----------	|
>| Non-Polyphase_T_16k_32k_I_16k_32k_LSD  	| 0.145649366 	| 3.88E-07 	| 0.209755 	|
>| Non-Polyphase_T_4k_8k_I_16k_32k_LSD    	| 0.020665331 	| 5.24E-07 	| 0.215294 	|
>| Non-Polyphase_T_16k_32k_I_16k_32k_WMSE 	| 0.002238482 	| 2.95E-07 	| 0.204589 	|
>| Non-Polyphase_T_4k_8k_I_16k_32k_WMSE   	| 0.10929662  	| 5.00E-07 	| 0.236057 	|
>| Polyphase_T_16k_32k_I_16k_32k_LSD      	| 0.134671843 	| 4.64E-07 	| 0.234291 	|
>| Polyphase_T_4k_8k_I_16k_32k_LSD        	| 0.129595473 	| 5.10E-07 	| 0.25999  	|
>| Polyphase_T_16k_32k_I_16k_32k_WMSE     	| 0.099928898 	| 3.47E-07 	| 0.2472   	|
>| Polyphase_T_4kk_8k_I_16k_32k_WMSE      	| 0.001909916 	| 3.42E-07 	| 0.245304 	|


## Subjective Results
> For subjective evaluation, a MUSHRA test was conducted among 7 people, with average hearing capabilities. Mushra stands for Multi-Stimulus test with Hidden Reference and Anchor(MUSHRA). MUSHRA is used for comparing the audio quality of different audio samples. MUSHRA has an advantage over Mean Opinion Score(MOS), as it can deliver better results with fewer participants. A participant is asked to rate the reconstructed audio files on a scale of 0-100 while switching in-between the reference and reconstructed. We use a web application developed by International Audio Labs Erlangen for this test. Where we have a reference signal or target signal compared to the reconstructed samples, reconstructed samples perceptually closer to the reference signal must have a high score. Here we are comparing target/reference and generated samples at 16kHz via Non-Polyphase based model, Polyphase based model, and Base model. 
![Mushra Test](imgs/Mushra_test_pres.png "Mushra test")

> Observations :
> From the results of the MUSHRA test, we can observe that the median value of Gener-
ated NO POLY T 8k 16k I 8k 16k WMSE(79.1) is the highest among the rest of the recon-
structed audio samples. Followed by Generated NO POLY T 4k 16k I 8k 16 WMSE(78.9),
which is much better than the Base model score of 69.5. Hence the method using Non-
Polyphase based model with WMSE as an error function provides the best results. For LSD as
error function for Non-Polyphase based model had median values better than Polyphase based
methods. In general, Non-Polyphase based models were better than Polyphase based models.
In general, all the audio samples generated with the Non-Polyphase based model using WMSE
and LSD were ranked higher than the Base model. These results prove the enhancement of our
proposed model over the base model.