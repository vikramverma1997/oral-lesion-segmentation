# oral-lesion-segmentation
> A light weight model to automatically segment oral buccal mucosa lesions from digital pictures of oral cavity.
>> The dataset used is not publicly available yet.

In this project:
- Transfer learning is used because the dataset we had was small.
- Different model architectures including U-Net, U-Net3+, and MultiResUnet are trained. Out of all these, U-Net with a backbone of EfficientNetb3 gave the best results.
- We obtained a Dice coefficient of 0.759 and Jaccard coefficient of 0.615 on the test data.
