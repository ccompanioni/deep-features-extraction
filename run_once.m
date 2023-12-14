%% Run only once in the first time to compile mex files
%%
clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
%% Compiling MatConvNet
vl_compilenn;