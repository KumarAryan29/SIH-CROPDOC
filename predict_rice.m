function predictedLabel = predict_rice()
    clc;
    clear all;
    load('Rice_model.mat');
    testimage = imread('D:\My Folder\SIH\WEB\uploads\input.jpeg');
    testimage = imresize(testimage, [224 224]);
    predictedLabel = string(classify(trainedNetwork_1, testimage));
    fprintf('Predicted Label: %s\n', string(predictedLabel));
    imshow(testimage);
    title(['Predicted Label: ', char(predictedLabel)]);
end