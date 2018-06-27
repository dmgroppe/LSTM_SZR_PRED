% This script loads and epoched EEGLAB data set, computes IC activations,
% normalizes the activations, and then makes a couple plots to figure out
% what a good threshold might be for detecting blinks

%load('P3_1cleanf.set','-MAT');
sub=1;
in_fname=sprintf('P3_%dcleanf.set',sub);
fprintf('Loading %s\n',in_fname);
load(in_fname,'-MAT');

%% compute IC acts
[n_chan, n_tpt, n_epoch]=size(EEG.data);
acts=EEG.icaweights*EEG.icasphere*reshape(EEG.data,n_chan,n_tpt*n_epoch);
% normalize each IC activation
for c=1:n_chan
    cntr=median(acts(c,:));
    disper=iqr(acts(c,:));
   acts(c,:)=(acts(c,:)-cntr)/disper; 
end
acts=reshape(acts,n_chan,n_tpt,n_epoch);

for a=1:length(EEG.labels)
    if ~isempty(EEG.labels{a})
       fprintf('%d: %s\n',a,EEG.labels{a}); 
    end
end


%% Compute std of each epoch
stds=zeros(n_chan,n_epoch);
for c=1:n_chan
    for e=1:n_epoch
        stds(c,e)=std(acts(c,:,e)); 
    end
end
stds=log(stds);

%%
figure(2); 
for a=1:6
subplot(3,2,a);
hist(stds(a,:),100);
axis tight;
title(sprintf('IC%d: %s',a,EEG.labels{a}));
end

%%
id=2;
thresh=0.5;
x_ids=find(stds(id,:)>=thresh);
norm_ids=find(stds(id,:)<thresh);
fprintf('%f pptn of trials above thresh.\n',mean(stds(id,:)>=thresh));
figure(1); clf;
subplot(1,2,1);
plot(squeeze(acts(id,:,x_ids)),'r-'); hold on;
axis tight;
subplot(1,2,2);
plot(squeeze(acts(id,:,norm_ids)),'b-'); hold on;
axis tight;
title(sprintf('IC %d',id));

%%
id=2; %MiPf
id=1;
figure(1); clf;
plot(squeeze(EEG.data(id,:,:)));
title(EEG.chanlocs(id).labels);

%%
s=size(EEG.data);
for c=1:s(1),
    for ep=1:s(3),
        EEG.data(c,:,ep)=detrend(EEG.data(c,:,ep));
    end
end
disp('done detrending!');