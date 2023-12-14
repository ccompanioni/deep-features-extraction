clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
run vl_setupnn;
%%
dt = load('feat.mat');
%%
modelTypeG = 'imagenet-googlenet-dag';
netG = load(['models/' modelTypeG '.mat']);
modelTypeR = 'imagenet-resnet-152-dag';
netR = load(['models/' modelTypeR '.mat']);
modelTypeV = 'imagenet-vgg-verydeep-19';
netV = load(['models/' modelTypeV '.mat']);
%%
disp('CNN Feature Extraction');
qryDir = './pics/query/';
fPath = [qryDir '74.jpg'];
cnnFeatG = getCnnFeatures(fPath,modelTypeG,netG);
cnnFeatR = getCnnFeatures(fPath,modelTypeR,netR);
cnnFeatV = getCnnFeatures(fPath,modelTypeV,netV);
cnnFeatA = [cnnFeatG,cnnFeatR,cnnFeatV];
%%
dst_term = 'cosine';
dst_dpG = pdist2(dt.cnnFeatG, cnnFeatG, dst_term);
dst_dpR = pdist2(dt.cnnFeatR, cnnFeatR, dst_term);
dst_dpV = pdist2(dt.cnnFeatV, cnnFeatV, dst_term);
dst_dpA = pdist2(dt.cnnFeatA, cnnFeatA, dst_term);