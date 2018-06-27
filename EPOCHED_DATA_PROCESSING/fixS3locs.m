subs=1:8;
for sub=subs,
    fname=['P6_' int2str(sub) 'cleanf.set']; 
    %fname=['P6_' int2str(sub) 'cleanf_short.set'];
    load(fname,'-MAT');
    
    fprintf('Fixing: %s\n',fname);
    
    cnt=0;
    for a=1:length(EEG.chanlocs),
    mtch=0;
    if strcmpi(EEG.chanlocs(a).labels,'lhe')
        EEG.chanlocs(a).labels='lhz';
    elseif strcmpi(EEG.chanlocs(a).labels,'rhe')
        EEG.chanlocs(a).labels='rhz';
    end
    for b=1:length(locs)
     if (strcmp(locs(b).labels,EEG.chanlocs(a).labels))
      fprintf('Match: %s\n',locs(b).labels);   
      cnt=cnt+1;
      EEG.chanlocs(a).theta=locs(b).theta;
      EEG.chanlocs(a).radius=locs(b).radius;
      EEG.chanlocs(a).labels=locs(b).labels;
      EEG.chanlocs(a).sph_theta=locs(b).sph_theta;
      EEG.chanlocs(a).sph_phi=locs(b).sph_phi;
      EEG.chanlocs(a).sph_radius=locs(b).sph_radius;
      %EEG.chanlocs(a).sph_theta_besa=locs(b).sph_theta_besa;
      EEG.chanlocs(a).X=locs(b).X;
      EEG.chanlocs(a).Y=locs(b).Y;
      EEG.chanlocs(a).Z=locs(b).Z;      
      mtch=1;
     end
    end
     if ~mtch,
        fprintf('MISMATCH: %s\n',EEG.chanlocs(a).labels); 
     end
    end
    cnt
    save(fname,'EEG');
end