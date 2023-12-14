clc; close all; clear all; warning off;
%%
imgDir = './NTB/dataset/';
%%
imgSets = imageSet(imgDir, 'recursive');
imageNames = {};
labels = {};
catNum = [];
itr=1;
for i=1:length(imgSets) 
    labels{i} = imgSets(i).Description;
    for j=1:length(imgSets(i).ImageLocation)
        [~,fname,ext] = fileparts(imgSets(i).ImageLocation{j});
        imageNames{itr} = [fname,ext];
        catNum(itr) = i;
        itr = itr+1;
    end
end
%%
save('dbMeta.mat','imageNames','labels','catNum');