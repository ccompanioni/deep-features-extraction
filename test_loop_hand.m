clc; close all; clear all; warning off;
%%
restoredefaultpath;
% addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
% addpath(genpath(fullfile('./libs/','vlfeat-0.9.20')));
% vl_setup;
%%
imgDir = './data/dataset/';
%%
disp('Hand Feature Extraction');
imagefiles = dir([imgDir '*.jpg']);      
num = length(imagefiles); 
feature_descriptors={};
all_desriptors = [];
cellSize = 8;
for i=1:num
    fname = imagefiles(i).name;   
    disp(['Processing ... ' num2str(i) ' of ' num2str(num) ' : ' fname]);
    fPath = [imgDir fname];
    img = imread(fPath);
%     feat = vl_hog(single(rgb2gray(img)),cellSize) ;
%     feat = vl_lbp(single(rgb2gray(img)),cellSize) ;
%     feat = reshape(feat,[size(feat,1)*size(feat,2) size(feat,3)]);
    feat = extractLBPFeatures(single(rgb2gray(img)),'CellSize',[cellSize,cellSize]);
    feat = reshape(feat,[],59);
    feature_descriptors{i}=feat;
    all_desriptors = [all_desriptors,feat'];
end
%%
disp('Hand BoW Clustering');
num_coodBook = 500;
all_desriptors=single(all_desriptors);
% [coodBook,cluster_idx]=vl_kmeans(all_desriptors,num_coodBook,'distance','l2','algorithm','ElKAN');
% coodBook = coodBook';

[coodBook,cluster_idx] = kmeans(all_desriptors',num_coodBook);
temp = coodBook;
coodBook = cluster_idx;
cluster_idx = temp;

% save('data.mat');
%%
% load('data.mat');

disp('Hand BoW Histogram');
histograms = zeros(num,num_coodBook);
K=1;
for i=1:num
    fname = imagefiles(i).name;
    disp(['Processing ... ' num2str(i) ' of ' num2str(num) ' : ' fname]);
    feature=feature_descriptors{i};
    % Histogram calculation
    class=knnclassify(feature,coodBook,1:size(coodBook,1),K,'euclidean','nearest');
    histcal=zeros(1,size(coodBook,1));
    for q=1:size(class,1)
        histcal(1,class(q))=histcal(1,class(q))+1;
    end
    % Histogram normalization 
    histcal= histcal/norm(histcal);
    histograms(i,:)= histcal;
end
%%
BoW.codeBook = coodBook;
BoW.K = K;
save('featHandLbp2.mat','imagefiles','histograms','BoW');