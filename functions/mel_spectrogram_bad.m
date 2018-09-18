%(c) 2018, Ivan Himawan, Queensland University of Technology.
% Acknowledgement:
% [1] M. Sahidullah, T. Kinnunen and C. Hanilci; "A comparison of features for synthetic speech detection"; Proc. Interspeech 2015, pp. 2087-2091, Dresden, Germany, September 2015.

function [feats] = mel_spectrogram_bad(X, fs)

% CONSTANTS ===============================================================
window_length = 20/1000;
D = window_length*fs;
L = 2^nextpow2(D);
filter_length = L;
SP = 0.5; % Overlap factor
No_Filter = 80;
%==========================================================================

% Segments -> no_frames x Length  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sg = buffer(X, D, ceil(SP*D), 'nodelay').';

% Windowing and FFT
no_frames = size(sg,1);
window = repmat(hamming(D).',no_frames,1);

y_framed = ((sg.*window));

% -------------------------------------------------------------------------

f = (fs/2) * linspace(0,1,filter_length/2+1);
fmel = 2595 * log10(1+f./700); % Converting to Mel-scale
fmelmax = max(fmel);
fmelmin = min(fmel);

filbandwidthsmel=linspace(fmelmin,fmelmax,No_Filter+2);
filbandwidthsf=700*(10.^(filbandwidthsmel/2595)-1);
fr_all=(abs(fft(y_framed',filter_length))).^2;
fa_all=fr_all(1:(filter_length/2)+1,:)';
filterbank=zeros((filter_length/2)+1,No_Filter);
for i=1:No_Filter
    filterbank(:,i)=trimf(f,[filbandwidthsf(i),filbandwidthsf(i+1),...
        filbandwidthsf(i+2)]);
end
filbanksum=fa_all*filterbank(1:end,:);

feats = log(filbanksum+eps); % frames x number of components

% imagesc(flipud(feats'))

