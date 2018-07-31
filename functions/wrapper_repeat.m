function [feats] = wrapper_repeat(X, fs)

t = 1;
for n = 1:2:10                                                                                                           
   seg = X(1+(n-1)*fs:(n+1)*fs);                                                                                               
   feats(:,:,t) = mel_spectrogram_bad([seg;0], fs);
   t = t + 1;

end
