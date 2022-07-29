Description :
In all the folders mentioned above, we have generated audio files, target and input audio files
4k-8k
8k-16k
16k-32k

1. Generated_(NO_POLY or POLY)_T_XXk_YYk_I_XXk_YYk_(WMSE or LSD).wav
   1. NO_POLY   :  No Polyphase Input method
   2. POLY      :  Polyphase method
   3. T_XXk_YYk :  Meaning trained for input sampled at XX kHz and target is sampled at YY kHz
   4. I_XXk_YYk :  Meaning Inferred for input sampled at XX kHz and predicted is sampled at YY kHz 
   5. WMSE      :  Weighted Mean Square as error function while training
   6. LSD       :  Log Spectral Distance as error function while training