clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
run vl_setupnn;
%%
modelTypeG = 'imagenet-googlenet-dag';
netG = load(['models/' modelTypeG '.mat']);
modelTypeR = 'imagenet-resnet-152-dag';
netR = load(['models/' modelTypeR '.mat']);
modelTypeV = 'imagenet-vgg-verydeep-19';
netV = load(['models/' modelTypeV '.mat']);
%%
%%
imgDir = './data/dataset/';
imagefiles = dir([imgDir '*.jpg']);      
num = length(imagefiles); 
cnnFeatG=[];cnnFeatR=[];cnnFeatV=[];cnnFeatD=[];
disp('CNN Feature Extraction');
for i=1:num
    fname = imagefiles(i).name;   
    disp(['Processing ... ' num2str(i) ' of ' num2str(num) ' : ' fname]);
    fPath = [imgDir fname];
    cnnFeatG(i,:) = getCnnFeatures(fPath,modelTypeG,netG);
    cnnFeatR(i,:) = getCnnFeatures(fPath,modelTypeR,netR);
    cnnFeatV(i,:) = getCnnFeatures(fPath,modelTypeV,netV);
    cnnFeatD(i,:) = [cnnFeatG(i,:),cnnFeatR(i,:),cnnFeatV(i,:)];
end
%%
save('featDeep.mat','imagefiles','cnnFeatG','cnnFeatR','cnnFeatV','cnnFeatD');