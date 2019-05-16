annoDir = '/home/samik/mnt/bnb/nfs/mitraweb2/mnt/disk125/main/MorseSkeleton_OSUMITRA/405_Results/405_ann';
imgDir = '/home/samik/mnt/bnb/nfs/mitraweb2/mnt/disk125/main/MorseSkeleton_OSUMITRA/405_Results/405_imgs';
dmDir = '/home/samik/mnt/bnb/nfs/mitraweb2/mnt/disk125/main/MorseSkeleton_OSUMITRA/405_Results/405_dimo_no_simp';
dm2Dir = '/home/samik/mnt/bnb/nfs/mitraweb2/mnt/disk125/main/MorseSkeleton_OSUMITRA/405_Results/405_simp_v2';
direc = dir(fullfile(annoDir, '*.tif'));
mkdir('P');
mkdir('P/405_ann/');
mkdir('P/405_imgs/');
mkdir('P/405_dimo_no_simp/');
mkdir('P/405_simp_v2/');


parfor d = 1 : length(direc)
    disp(d)
    imgAnno = imread(fullfile(annoDir, direc(d).name));
    img = imread(fullfile(imgDir, direc(d).name));
    imgV1 = imread(fullfile(dmDir, [direc(d).name(1:end-4) '.png']));
    imgV2 = imread(fullfile(dm2Dir, [direc(d).name(1:end-4) '.png']));
    for i = 1 : 256 : size(img,1)
        for j = 1 : 256 : size(img,2)
            if (i+255)<=size(img,1) && (j+255)<=size(img,2)
                patch1 = imgAnno(i:i+255,j:j+255);
                if sum(sum(patch1)) % Check if there are any signals in GT
                    patch2 = img(i:i+255,j:j+255);
                    patch3 = imgV1(i:i+255,j:j+255);
                    patch4 = imgV2(i:i+255,j:j+255);
                    imwrite(patch1, ['P/405_ann/' direc(d).name(1:end-4) '_' num2str(i) '_' num2str(j) '.tif']);
                    imwrite(patch2, ['P/405_imgs/' direc(d).name(1:end-4) '_' num2str(i) '_' num2str(j) '.tif']);
                    imwrite(patch3, ['P/405_dimo_no_simp/' direc(d).name(1:end-4) '_' num2str(i) '_' num2str(j) '.tif']);
                    imwrite(patch4, ['P/405_simp_v2/' direc(d).name(1:end-4) '_' num2str(i) '_' num2str(j) '.tif']);
                end
            end
        end
    end
end