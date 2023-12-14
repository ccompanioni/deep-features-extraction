clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
%% sample images
fPath = './input/tc4673af.jpg';
% fPath = './input/tc467231.jpg';
%%
% modelType = 'imagenet-googlenet-dag';
% modelType = 'imagenet-resnet-152-dag';
modelType = 'imagenet-vgg-verydeep-19';
net = load(['models/' modelType '.mat']);
run vl_setupnn;
%%
cnnFeat = getCnnFeatures(fPath,modelType,net);