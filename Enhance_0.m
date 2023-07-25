% Set the directory path
directory = 'F:\GradProj\IIIIII\data\4';

% Get a list of all image files in the directory
fileList = dir(fullfile(directory, '*.jpg')); % Modify the extension if necessary

% Loop through each image file
for i = 1:length(fileList)
    % Read the image
    imagePath = fullfile(directory, fileList(i).name);
    image = imread(imagePath);
    
   % Preprocessing steps
    % Convert to grayscale
    grayImage = rgb2gray(image);
    
    % Resize the image (optional)
    % resizedImage = imresize(grayImage, [desiredHeight, desiredWidth]);
    
    % Normalize the pixel values to [0, 1]
    normalizedImage = double(grayImage) / 255.0;
    
    % Denoising
    % Apply a denoising algorithm such as median filtering or Gaussian filtering
    denoisedImage = medfilt2(normalizedImage); % Example: Median filtering
    
    % Contrast enhancement
    % Enhance the contrast using histogram equalization
    enhancedImage = histeq(denoisedImage);
    
    % Sharpening (optional)
    % Apply image sharpening techniques like unsharp masking or the Laplacian filter
    sharpenedImage = imsharpen(enhancedImage); % Example: Unsharp masking
    
    % Augmentation (if desired)
    % Apply augmentation techniques such as rotation, scaling, flipping, etc.
    % augmentedImage = performAugmentation(sharpenedImage); % Example: Custom augmentation function
    
    % Save the enhanced image
    enhancedImagePath = fullfile(directory, ['enhanced_', fileList(i).name]);
    imwrite(enhancedImage, enhancedImagePath);
    
    % Display the original and enhanced images
    % figure;
    % subplot(1, 2, 1);
    % imshow(image);
    % title('Original Image');
    
    % subplot(1, 2, 2);
    % imshow(enhancedImage);
    % title('Enhanced Image');
end
