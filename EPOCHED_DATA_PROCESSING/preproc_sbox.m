%%
load('/home/dgroppe/GIT/LSTM_SZR_PRED/EPOCHED_DATA/DATA_ODBL30/P3_1cleanf.set','-MAT');
orig_data=EEG.data;

%%
[n_chan, n_tpt, n_ep]=size(EEG.data);
for a=1:n_ep
    EEG.data(:,:,a)=butterfilt3(squeeze(double(EEG.data(:,:,a))),EEG.srate,[1 EEG.srate/2],0);
end

%   Usage:
%    >> data=butterfiltMK(data,srate,flt,n_pad);
%  
%   Required Inputs:
%     data      - 2 dimensional (channel x time point) matrix of EEG data
%     srate     - Sampling rate (in Hz)
%     filt      - [low_boundary high_boundary] a two element vector indicating the frequency
%                 cut-offs for a 3rd order Butterworth filter that will be applied to each
%                 trial of data.  If low_boundary=0, the filter is a low pass filter.  If
%                 high_boundary=(sampling rate)/2, then the filter is a high pass filter.  If both
%                 boundaries are between 0 and (sampling rate)/2, then the filter is a band-pass filter.
%                 If both boundaries are between 0 and -(sampling rate)/2, then the filter is a band-
%                 stop filter (with boundaries equal to the absolute values of the low_boundary and
%                 high_boundary).  Note, using this option requires the signal processing toolbox
%                 function butter.m.  You should probably use the 'baseline' option as well since the
%                 mean prestimulus baseline may no longer be 0 after the filter is applied

%%
id=ceil(rand()*n_ep);
figure(1); clf;
plot(EEG.data(1,:,id),'b-'); hold on;
plot(orig_data(1,:,id),'r-');
title(sprintf('Epoch %d',id));
legend('High Pass','Orig');