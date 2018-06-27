%%
% one subject is missing 'lle'
%use_elecs=['lle','lhz','rhz','LLpf'
use_elecs={'LLpf','MiPf','RLPf','LLFr','LDFr','RDFr','RLFr', ...
    'MiCe','MiPa','MiOc'};
n_use_elec=length(use_elecs);

high_pass=1;

raw_eeg_cell=cell(1,1);
blink_eeg_cell=cell(1,1);
blink_class_cell=cell(1,1);
n_group_epochs=0;



%%
subs=1:8;
for sub=subs
    % load data
    in_fname=sprintf('P3_%dcleanf.set',sub);
    fprintf('Loading %s\n',in_fname);
    load(in_fname,'-MAT');
    
    use_elec_ids=zeros(n_use_elec,1);
    for a=1:n_use_elec
       for c=1:length(EEG.chanlocs)
          if strcmpi(EEG.chanlocs(c).labels,use_elecs{a})
             use_elec_ids(a)=c;
             break;
          end
       end
       if use_elec_ids(a)==0
          error('Could not find electrode %s in this EEG struct',use_elecs{a}); 
       end
    end
    
    % Find blink IC
    blink_ic=0;
    for a=1:length(EEG.labels)
        if strcmpi(EEG.labels{a},'blink')
            blink_ic=a;
            break;
        end
    end
    if blink_ic==0
        error('Could not find blink IC\n');
    else
        fprintf('blink IC is %d\n',blink_ic);
    end
    
    % copmute IC activations
    [n_chan, n_tpt, n_epoch]=size(EEG.data);
    %unmix=inv(EEG.icawinv);
    %acts=EEG.icaweights*EEG.icasphere*reshape(EEG.data,n_chan,n_tpt*n_epoch);
    %acts=unmix*reshape(EEG.data,n_chan,n_tpt*n_epoch);
    acts=EEG.icawinv\reshape(EEG.data,n_chan,n_tpt*n_epoch);
    blink_scalp=reshape(EEG.icawinv(:,blink_ic)*acts(blink_ic,:),n_chan,n_tpt, ...
        n_epoch);
    
    % normalize each IC's activation
    for c=1:n_chan
        cntr=median(acts(c,:));
        disper=iqr(acts(c,:));
        acts(c,:)=(acts(c,:)-cntr)/disper;
    end
    acts=reshape(acts,n_chan,n_tpt,n_epoch);
        
    %% Compute std of each epoch
    stds=zeros(n_chan,n_epoch);
    for c=1:n_chan
        for e=1:n_epoch
            stds(c,e)=std(acts(c,:,e));
        end
    end
    stds=log(stds);
    
    % identify epochs with blinks
    thresh=0.5;
    blink_ids=find(stds(blink_ic,:)>=thresh);
    norm_ids=find(stds(blink_ic,:)<thresh);
    n_blink=length(blink_ids);
    fprintf('%f pptn of trials above thresh.\n',n_blink/n_epoch);
    
    % randomly choose an equal # of non-blink epochs
    non_blink_ids=setdiff(1:n_epoch,blink_ids);
    perm_ids=randperm(n_epoch-n_blink);
    use_non_blink_ids=non_blink_ids(perm_ids(1:n_blink));
    
    % for those epochs (at select channels) collect 
    % raw eeg data
    raw_eeg_cell{sub}=EEG.data(use_elec_ids,:,[blink_ids use_non_blink_ids]);
    
    % scalp projected blinks
    blink_eeg_cell{sub}=blink_scalp(use_elec_ids,:,[blink_ids use_non_blink_ids]);
    
    blink_class_cell{sub}=zeros(n_blink*2,1);
    blink_class_cell{sub}(1:n_blink)=1;
    n_group_epochs=n_group_epochs+n_blink*2;
end

%% Combine all data into a single matrices
raw_eeg=zeros(n_use_elec,n_tpt,n_group_epochs);
blink_eeg=raw_eeg;
blink_class=zeros(n_group_epochs,1);
sub_ids=zeros(n_group_epochs,1);

cursor=1;
for sub=subs
    n_ep_this_sub=size(raw_eeg_cell{sub},3);
    raw_eeg(:,:,cursor:cursor+n_ep_this_sub-1)=raw_eeg_cell{sub};
    blink_eeg(:,:,cursor:cursor+n_ep_this_sub-1)=blink_eeg_cell{sub};
    blink_class(cursor:cursor+n_ep_this_sub-1,1)=blink_class_cell{sub};
    sub_ids(cursor:cursor+n_ep_this_sub-1,1)=sub;
    cursor=cursor+n_ep_this_sub;
end

% create artifact corrected data
cleaned_eeg=raw_eeg-blink_eeg;
srate=EEG.srate;
% save to disk
save('p3_epoched_blink_group.mat','raw_eeg','blink_eeg','cleaned_eeg','blink_class', ...
    'use_elecs','srate');

%%
figure(1); clf; imagesc(squeeze(raw_eeg(1,:,:))); title('Raw EEG, Chan 1'); colorbar;
figure(2); clf; imagesc(squeeze(blink_eeg(1,:,:))); title('Blink EEG, Chan 1'); colorbar;
figure(3); clf; imagesc(squeeze(cleaned_eeg(1,:,:))); title('Cleaned EEG, Chan 1'); colorbar;
figure(4); clf; plot(sub_ids); hold on; plot(blink_class,'--'); colorbar;

