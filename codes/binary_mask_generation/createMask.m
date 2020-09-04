function r = createMask(main_dir)

% main_dir = 'images_marked/';
mask_dir = 'Mask_binary';

% Verifies if the 'Mask_binary' dir exists
if ~exist(strcat(main_dir, mask_dir), 'dir')
    mkdir(strcat(main_dir, mask_dir))
end

% Saves a variable 'files' with all the names of .jpg files
files = dir(fullfile(strcat(main_dir,'images_marked/'), '*.jpg'));
% Gets the number of files
no_files = size(files, 1);

for n=1:1:no_files
    
    %Variable with name of file
    nam = char(files(n).name);
    % Trims the extention of the file name
    ln_nm = size(nam, 2)-4;
    nam = nam(1, 1:ln_nm);
    % Open image
    img = imread(strcat(main_dir, 'images_marked/', nam, '.jpg'));
    img1 = imadjust(img,[0 0 0; 0.03 0.03 .03],[]);
    img_gray = rgb2gray(img1);
    imbw = imbinarize(img_gray,'adaptive', 'ForegroundPolarity','dark','Sensitivity',0.2);
    imbw = uint8(bwareaopen(imbw, 2000000));
    imbw1 = imfill(~imbw,'holes');
    imbw1 = imclearborder(imbw1);
    img_filt = filter_im(~imbw1);
    imgcomp = ~img_filt;
    
    %     figure
    %     subplot(1,2,1)
    %     imshow(imbw1)
    
    stats = regionprops(imgcomp,'BoundingBox','Circularity','Area');
    %     hold on
    m = length(stats);
    
    if m == 1
        %         BB = stats.BoundingBox;
        %         rectangle('Position',[BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2);
        imgmask = imgcomp;
    else
        
        for k = 1:m
            %             BB = stats(k).BoundingBox;
            %             rectangle('Position',[BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2);
            round(k) = stats(k).Circularity < 0.25;
        end
        
        I = find(round);
        
        if isempty(I)
            imgmask = imgcomp;
        else
            
            [~, y] = size(I);
            area_lesion = [];
            for n = 1:y
                area_lesion(1,n) = stats(I(n)).Area;
            end
            
            imgmask = bwpropfilt(imgcomp,'Area',[max(area_lesion(:))+1,Inf]);
        end
    end
    
    windowSize = 51;
    kernel = ones(windowSize) / windowSize ^ 2;
    blurryImage = conv2(single(imgmask), kernel, 'same');
    binaryImage = blurryImage > 0.647;
    % Saves the final image with th '_lic' sufix
    imwrite(binaryImage, strcat(main_dir, mask_dir, '\', '\', nam, '_mask', '.jpg'));
    clear round;
end
r = 1;
end