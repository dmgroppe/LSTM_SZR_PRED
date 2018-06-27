subs=[1 2 4 5 6 7 8];
subs=1:8;
plotem=0;

for sub=subs,
  fname=['P3_' int2str(sub) 'cleanf.set'];
  load(fname,'-MAT');
  
  sublocs{sub}=EEG.chanlocs;
  
  Nchans=size(EEG.data,1);
  Nepochs=size(EEG.data,3);
  Npoints=size(EEG.data,2);
  
  data=rmbase(EEG.data,Npoints,1:25);
  %remove artifacts
  art_ics=[];
  for i=1:size(EEG.labels,2),
    if (isart(EEG.labels(i)))
        art_ics=[art_ics i];
    end;
  end;
  mix=EEG.icawinv;
  unmix=EEG.icaweights*EEG.icasphere;
  EEG.data=remove_comps(EEG.data,unmix,mix,art_ics);

  p3Txo=[];
  p3Sxo=[];
  p3Tcase=[];
  p3Scase=[];
  p3Tcat=[];
  p3Scat=[];
  for a=1:Nepochs,
      if (EEG.event(a).XOtarg)
         p3Txo=[p3Txo a];
      elseif (EEG.event(a).XOstand)
         p3Sxo=[p3Sxo a];
      elseif (EEG.event(a).CASEstand)
         p3Scase=[p3Scase a];
      elseif (EEG.event(a).CASEtarg)
         p3Tcase=[p3Tcase a]; 
      elseif (EEG.event(a).CATstand)
         p3Scat=[p3Scat a];
      elseif (EEG.event(a).CATtarg)
         p3Tcat=[p3Tcat a]; 
      end
  end

  %XO targ & stand ERP
  dt=EEG.data(:,:,p3Txo);
  dt=rmbase(dt,Npoints,1:25);
  tXO{sub}=blockave(dt,249);
  
  dt=EEG.data(:,:,p3Sxo);
  dt=rmbase(dt,Npoints,1:25);
  sXO{sub}=blockave(dt,249);
  
  if plotem,
     plot4topo(tXO{sub},sXO{sub},EEG.chanlocs,[-.1 .892],[.170 .28 .5 .7], ...
      ['Sub ' int2str(sub) ':XO targ & stand'],-.1, ...
      'Targ','Stand');  
  end
  
  
  %CASE targ & stand ERP
  dt=EEG.data(:,:,p3Tcase);
  dt=rmbase(dt,Npoints,1:25);
  tCASE{sub}=blockave(dt,249);
  
  dt=EEG.data(:,:,p3Scase);
  dt=rmbase(dt,Npoints,1:25);
  sCASE{sub}=blockave(dt,249);
  
  if plotem,
     plot4topo(tCASE{sub},sCASE{sub},EEG.chanlocs,[-.1 .892],[.170 .28 .5 .7], ...
      ['Sub ' int2str(sub) ':CASE targ & stand'],-.1, ...
      'Targ','Stand');  
  end
  
  
  %CAT targ & stand ERP
  dt=EEG.data(:,:,p3Tcat);
  dt=rmbase(dt,Npoints,1:25);
  tCAT{sub}=blockave(dt,249);
  
  dt=EEG.data(:,:,p3Scat);
  dt=rmbase(dt,Npoints,1:25);
  sCAT{sub}=blockave(dt,249);

  if plotem,
     plot4topo(tCAT{sub},sCAT{sub},EEG.chanlocs,[-.1 .892],[.170 .28 .5 .7], ...
      ['Sub ' int2str(sub) ':CAT targ & stand'],-.1, ...
      'Targ','Stand');  
  end

end;

save indiv_erpsODBL30 sCASE tCASE sCAT tCAT sXO tXO sublocs
