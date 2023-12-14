clc; close all; clear all; warning off;
%%
restoredefaultpath;
addpath(genpath(fullfile('./libs/','matconvnet-1.0-beta24')));
run vl_setupnn;
%%
% qtype: 1=athlete, 2=football, 3=indoor, 4=outdoor, 
%        5=portrait, 6=ski
load('dbMeta.mat');
dtD = load('featDeep.mat');
dtH = load('featHandLbp2.mat');
qryDir = './data/query/';
imgDir = './data/dataset/';
qname = 'tb99b389.jpg'; %%% INPUTv 
qtype = 6; %%% INPUT
qPath = [qryDir qname];
%%
modelTypeG = 'imagenet-googlenet-dag';
netG = load(['models/' modelTypeG '.mat']);
modelTypeR = 'imagenet-resnet-152-dag';
netR = load(['models/' modelTypeR '.mat']);
modelTypeV = 'imagenet-vgg-verydeep-19';
netV = load(['models/' modelTypeV '.mat']);
%%
disp('CNN Feature Extraction');
% tic;
cnnFeatG = getCnnFeatures(qPath,modelTypeG,netG);
% toc;
% tic;
cnnFeatR = getCnnFeatures(qPath,modelTypeR,netR);
% toc;
% tic;
cnnFeatV = getCnnFeatures(qPath,modelTypeV,netV);
% toc;
cnnFeatD = [cnnFeatG,cnnFeatR,cnnFeatV];
%%
% addpath(genpath(fullfile('./libs/','vlfeat-0.9.20')));
% vl_setup;
%%
disp('Hand Feature Extraction');
cellSize = 8;
img = imread(qPath);
% feat = vl_lbp(single(rgb2gray(img)),cellSize) ;
% feat = reshape(feat,[size(feat,1)*size(feat,2) size(feat,3)]);
feat = extractLBPFeatures(single(rgb2gray(img)),'CellSize',[cellSize,cellSize]);
feat = reshape(feat,[],59);
% save('data2.mat');
% return;
% load('data2.mat');
% Histogram calculation
class=knnclassify(feat,dtH.BoW.codeBook,1:size(dtH.BoW.codeBook,1),dtH.BoW.K,'euclidean','nearest');
histcal=zeros(1,size(dtH.BoW.codeBook,1));
for q=1:size(class,1)
    histcal(1,class(q))=histcal(1,class(q))+1;
end
% Histogram normalization 
histcal= histcal/norm(histcal);
%%
dst_term = 'euclidean';
% dst_dpG = pdist2(dtD.cnnFeatG, cnnFeatG, dst_term);
% dst_dpR = pdist2(dtD.cnnFeatR, cnnFeatR, dst_term);
% dst_dpV = pdist2(dtD.cnnFeatV, cnnFeatV, dst_term);
dst_dpD = pdist2(dtD.cnnFeatD, cnnFeatD, dst_term);
dst_dpD = 1-dst_dpD./max(dst_dpD);
dst_dpH = pdist2(dtH.histograms, histcal, dst_term);
dst_dpH = 1-dst_dpH./max(dst_dpH);
dst_dpDH = pdist2([dtD.cnnFeatD,dtH.histograms], [cnnFeatD,histcal], dst_term);
dst_dpDH = 1-dst_dpDH./max(dst_dpDH);
%%
[dst,idx] = sort(dst_dpD,'descend');   

% figure;
% subplot(2, 3, 1);
% imshow(imread(qPath),[]); axis off;
% title(['Query - Deep (' labels{qtype} ')']);
% for i=1:min(5,numel(idx))
%     subplot(2, 3, i+1); 
%     imshow(imread([imgDir dtD.imagefiles(idx(i)).name]),[]);
%     fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name));
%     title([num2str(dst(i)) ' (' labels{catNum(fIdx)} ')']);
%     axis off;
% end

DBtype = zeros(1,numel(idx));
for i=1:numel(idx)
    fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name),1);
    DBtype(i) = catNum(fIdx);
end

targets = ismember(DBtype,qtype);
relevant_IDs = find(targets==1);
relevant_IDs = sort(relevant_IDs);
numRel = numel(relevant_IDs);
pre = (1:numRel) ./ relevant_IDs;
rec = (1:numRel) / numRel;
% figure;
% plot(rec, pre, 'b.-');
% xlabel('rec');
% ylabel('pre');
% title('pre-rec Curve - Deep');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

numNonRel = sum(targets==0);
fpr = zeros(1,numel(rec));
for i=1:numel(rec)
    fpr(i)= (relevant_IDs(i)-i)/numNonRel;
end

drec11 = 1:-0.1:0;
dpre11 = zeros(1,11);
dfpr11 = zeros(1,11);
for i=1:numel(drec11)
   dpre11(i) = max(pre((rec >=drec11(i))));
   dfpr11(i) = min(fpr((rec >=drec11(i))));
end

figure;
plot(drec11, dpre11, 'b.-');
xlabel('rec');
ylabel('pre');
title('pre-rec Curve (11-Points) - Deep');
axis([0 1 0 1.05]); %// Adjust axes for better viewing
grid;
% 
% figure;
% plot(fpr11, rec11, 'b.-');
% xlabel('FPR');
% ylabel('tpr/rec');
% title('ROC Curve (11-Points) - Deep');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

targetsTop = targets(1:5);
relevant_IDsTop = find(targetsTop==1);
preTop = numel(relevant_IDsTop) / 5;
recTop = numel(relevant_IDsTop) / numRel;

display(['Top 5 - Deep: AUC of ROC curve = ' num2str(trapz(fpr, rec))]);

display(['Top 5 - Deep: Rec = ' num2str(recTop) ...
    ', Pre = ' num2str(preTop)]);
%%
[dst,idx] = sort(dst_dpH,'descend');   

% figure;
% subplot(2, 3, 1);
% imshow(imread(qPath),[]); axis off;
% title(['Query - Hand (' labels{qtype} ')']);
% for i=1:min(5,numel(idx))
%     subplot(2, 3, i+1); 
%     imshow(imread([imgDir dtD.imagefiles(idx(i)).name]),[]);
%     fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name));
%     title([num2str(dst(i)) ' (' labels{catNum(fIdx)} ')']);
%     axis off;
% end

DBtype = zeros(1,numel(idx));
for i=1:numel(idx)
    fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name),1);
    DBtype(i) = catNum(fIdx);
end

targets = ismember(DBtype,qtype);
relevant_IDs = find(targets==1);
numRel = numel(relevant_IDs);
pre = (1:numRel) ./ relevant_IDs;
rec = (1:numRel) / numRel;
% figure;
% plot(rec, pre, 'b.-');
% xlabel('rec');
% ylabel('pre');
% title('pre-rec Curve - Hand');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

numNonRel = sum(targets==0);
fpr = zeros(1,numel(rec));
for i=1:numel(rec)
    fpr(i)= (relevant_IDs(i)-i)/numNonRel;
end

hrec11 = 1:-0.1:0;
hpre11 = zeros(1,11);
hfpr11 = zeros(1,11);
for i=1:numel(hrec11)
   hpre11(i) = max(pre((rec >=hrec11(i))));
   hfpr11(i) = min(fpr((rec >=hrec11(i))));
end

figure;
plot(hrec11, hpre11, 'b.-');
xlabel('rec');
ylabel('pre');
title('pre-rec Curve (11-Points) - Hand');
axis([0 1 0 1.05]); %// Adjust axes for better viewing
grid;
% 
% figure;
% plot(fpr11, rec11, 'b.-');
% xlabel('FPR');
% ylabel('tpr/rec');
% title('ROC Curve (11-Points) - Hand');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

targetsTop = targets(1:5);
relevant_IDsTop = find(targetsTop==1);
preTop = numel(relevant_IDsTop) / 5;
recTop = numel(relevant_IDsTop) / numRel;

display(['Top 5 - Hand: AUC of ROC curve = ' num2str(trapz(fpr, rec))]);

display(['Top 5 - Hand: Rec = ' num2str(recTop) ...
    ', Pre = ' num2str(preTop)]);
%%
[dst,idx] = sort(dst_dpDH,'descend');   

% figure;
% subplot(2, 3, 1);
% imshow(imread(qPath),[]); axis off;
% title(['Query - Deep + Hand (' labels{qtype} ')']);
% for i=1:min(5,numel(idx))
%     subplot(2, 3, i+1); 
%     imshow(imread([imgDir dtD.imagefiles(idx(i)).name]),[]);
%     fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name));
%     title([num2str(dst(i)) ' (' labels{catNum(fIdx)} ')']);
%     axis off;
% end

DBtype = zeros(1,numel(idx));
for i=1:numel(idx)
    fIdx = find(ismember(imageNames,dtD.imagefiles(idx(i)).name),1);
    DBtype(i) = catNum(fIdx);
end

targets = ismember(DBtype,qtype);
relevant_IDs = find(targets==1);
numRel = numel(relevant_IDs);
pre = (1:numRel) ./ relevant_IDs;
rec = (1:numRel) / numRel;
% figure;
% plot(rec, pre, 'b.-');
% xlabel('rec');
% ylabel('pre');
% title('pre-rec Curve - Deep + Hand');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

numNonRel = sum(targets==0);
fpr = zeros(1,numel(rec));
for i=1:numel(rec)
    fpr(i)= (relevant_IDs(i)-i)/numNonRel;
end

rec11 = 1:-0.1:0;
pre11 = zeros(1,11);
fpr11 = zeros(1,11);
for i=1:numel(rec11)
   pre11(i) = max(pre((rec >=rec11(i))));
   fpr11(i) = min(fpr((rec >=rec11(i))));
end

% figure;
% plot(rec11, pre11, 'b.-');
% xlabel('rec');
% ylabel('pre');
% title('pre-rec Curve (11-Points) - Deep + Hand');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

% figure;
% plot(fpr11, rec11, 'b.-');
% xlabel('FPR');
% ylabel('tpr/rec');
% title('ROC Curve (11-Points) - Deep + Hand');
% axis([0 1 0 1.05]); %// Adjust axes for better viewing
% grid;

targetsTop = targets(1:5);
relevant_IDsTop = find(targetsTop==1);
preTop = numel(relevant_IDsTop) / 5;
recTop = numel(relevant_IDsTop) / numRel;

display(['Top 5 - Deep + Hand: AUC of ROC curve = ' num2str(trapz(fpr, rec))]);

display(['Top 5 - Deep + Hand: Rec = ' num2str(recTop) ...
    ', Pre = ' num2str(preTop)]);