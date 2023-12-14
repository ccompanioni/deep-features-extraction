function cnnFeatures = getCnnFeatures(varargin)
%% Image Loading
try 
    inputImgName = varargin{1};
    im = imread(inputImgName) ;
catch
    disp('Error in input: image path');
    return;
end
% disp(['Input image : ' name]);
%% Network Configuration
modelType = varargin{2};
net = varargin{3};
normalization = net.meta.normalization;

if (~isempty(strfind(modelType, 'dag')))
  net = dagnn.DagNN.loadobj(net) ;
  net.conserveMemory = false;
  dag = true ;
else
  net.layers = net.layers(1:end-2);
  dag = false ;
end
%% Data Pre-processing
im = imresize(im, normalization.imageSize(1:2), 'bilinear') ;
%% CNN Feature Extraction using GoogleLeNet Model
im_ = single(im); 
for i=1:size(im_,3)
    if(size(normalization.averageImage,3)>1)
        im_(:,:,i) = im_(:,:,i) - normalization.averageImage(:,:,i) ;
    else
        im_(:,:,i) = im_(:,:,i) - normalization.averageImage(i) ;
    end
end

if dag
    net.eval({'data',im_}) ;
    cnnFeatures = squeeze(gather(net.vars(end-2).value)) ;
else
    res = vl_simplenn(net, im_) ;
    cnnFeatures = squeeze(gather(res(end).x)) ;
end
cnnFeatures = cnnFeatures' / norm(cnnFeatures);
