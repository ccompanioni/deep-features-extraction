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
dt = load([modelType '-feat.mat']);
%%
net = load(['models/' modelType '.mat']);
%%
disp('CNN Feature Extraction');
qryDir = './pics/query/';
fPath = [qryDir '74.jpg'];
cnnFeat = getCnnFeatures(fPath,modelType,net);
%%
dst_term = 'cosine';
dst_dp = pdist2(dt.cnnFeat, cnnFeat, dst_term);