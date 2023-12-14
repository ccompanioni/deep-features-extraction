clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
run vl_setupnn;
%%
modelType = 'imagenet-googlenet-dag';
% modelType = 'imagenet-resnet-152-dag';
% modelType = 'imagenet-vgg-verydeep-19';
%%
net = load(['models/' modelType '.mat']);
%%
imgDir = './pics/dataset/';
imagefiles = dir([imgDir '*.jpg']);      
num = length(imagefiles); 
cnnFeat=[];
disp('CNN Feature Extraction');
for i=1:num
    fname = imagefiles(i).name;   
    disp(['Processing ... ' num2str(i) ' of ' num2str(num) ' : ' fname]);
    fPath = [imgDir fname];
    cnnFeat(i,:) = getCnnFeatures(fPath,modelType,net);
end
%%
save([modelType '-feat.mat'],'imagefiles','cnnFeat');