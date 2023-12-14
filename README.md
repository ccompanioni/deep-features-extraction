# deep-features-extraction
Editorial Image Retrieval using Handcrafted and CNN Features

This work presents a combined feature representation based on handcrafted and deep approaches, to categorize editorial images into six classes (athletics, football, indoor, outdoor, portrait, and ski). Convolutional neural networks (CNNs) are used for image retrieval (NTB, the Norwegian News Agency). Deep features are extracted from three existing CNNs (VGG-VD, GoogLeNet, and ResNet) and fused to obtain image feature information. Hand-crafted features (LBP, BoW) are also combined to improve retrieval results.
Dataset: 17,000 medium-resolution images
Tools used: MATLAB, MatConvNet, K-means clustering, VLFeat

![Screenshot](cnn_query.png)

Users of this code are encouraged to cite the following article: Companioni-Brito, C., Elawady, M., Yildirim, S., Hardeberg, J.Y. (2018). Editorial Image Retrieval Using Handcrafted and CNN Features. In: Mansouri, A., El Moataz, A., Nouboud, F., Mammass, D. (eds) Image and Signal Processing. ICISP 2018. Lecture Notes in Computer Science(), vol 10884. Springer, Cham. https://doi.org/10.1007/978-3-319-94211-7_31
