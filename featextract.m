function run_featextract(filelist)

fileID = fopen(filelist);
tmp = textscan(fileID, '%s %s', 'Delimiter', ',');
fclose(fileID)

warning off;
% add required libraries to the path
addpath(genpath('functions'));
filenames = tmp{1};

c = 1;
for i = 1:length(filenames)
    [pathstr_w,name_w,ext_w] = fileparts(filenames{i});
    filenames{i}
    [x,fs] = audioread(filenames{i});
    x = x/sqrt(sum(abs(x.^2))/length(x));
    if (length(x) < 5*fs)
       dup = ceil(10*fs/length(x));
       x = repmat(x,dup,1);
    end
    %x = x/sqrt(sum(abs(x.^2))/length(x));
    x = [x;x(1:10*fs-length(x))];
    [feats] = wrapper_repeat(x,fs);
    myfeat{c} = reshape(feats(:,:,1),1,[]);
    myfeat{c+1} = reshape(feats(:,:,2),1,[]);
    myfeat{c+2} = reshape(feats(:,:,3),1,[]);
    myfeat{c+3} = reshape(feats(:,:,4),1,[]);
    myfeat{c+4} = reshape(feats(:,:,5),1,[]);
    c = c + 5;
end

C = strsplit(filelist,'/');
outfname = strcat(C{length(C)},'.mat')

save(outfname,'-v7.3','myfeat')

exit

