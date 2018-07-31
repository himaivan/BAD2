function [feats] = mel_spectrogram_bad(X, fs)

%clear; clc;

%wavfile = 'rainwav/20150624_064556_19204.wav';
%[X,fs] = audioread(wavfile); 

%st = randi(fs*57);
%X = X(st:st+fs*2);

% CONSTANTS ===============================================================
window_length = 20/1000; % 100 ms window
D = window_length*fs; % samples_block (no samples per block) = L
L = 2^nextpow2(D);
filter_length = L;
SP = 0.5; % Overlap factor
%inc = L - ceil(SP*L); % Number of advance samples
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

