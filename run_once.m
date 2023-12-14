%% Run only once in the first time to compile mex files
%%
clc; close all; clear all; warning off;
%%
restoredefaultpath;
%% Compiling MatConvNet
% addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
% vl_compilenn;
%% Compiling VLfeat
addpath(genpath(fullfile('./libs/','vlfeat-0.9.20')));
vl_setup;
vl_compile;