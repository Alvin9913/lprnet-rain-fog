# License Plate Recognition in Rain and Fog

A license plate model can score well on clear photos and still fail badly in bad weather. So in this project I first generate the bad weather myself, then train the model on it.

## What I did

1. Used `cv2` to degrade the plate images in two ways:
   - **Fog**: `addWeighted` to blend the image with a white layer, which lowers visibility
   - **Rain**: draw random rain streak lines on top, which partly covers the characters
2. Mixed the generated images into the CCPD2020 green-plate training set (**5769 images**) and kept the original validation set for comparison.
3. Retrained LPRNet and kept three checkpoints: 2000 iterations, 4000 iterations and the final model.

## What is in this repo

```
lprnet-rain-fog.ipynb   the weather synthesis, the data paths, and the training launch
test_LPRNet.py          inference script (load weights, read out the plate characters)
weights/
  LPRNet_iter_2000.pth
  LPRNet_iter_4000.pth
  Final_LPRNet_model.pth
```

## To be clear about what is mine

- The network and the training loop come from the open source project **LPRNet_Pytorch** (author aiboy.wei). What I did is the **weather synthesis, the data organisation, and running and keeping the training**. I did not write LPRNet from scratch.
- I ran the training from the command line, so the notebook does not contain a full per-epoch log. What can be checked is the three checkpoints and the size of the dataset.
- For the accuracy number: I would have to load the weights and run them on the CCPD2020 test set to get it, and I did not keep that number, so I am not writing one here.

## How to reproduce

```bash
pip install torch torchvision opencv-python numpy
# 1. get the CCPD2020 green plate data (train / val)
# 2. run the weather synthesis in the notebook to build the augmented training folder
# 3. point the upstream train_LPRNet.py at that folder, or load the weights and run inference
python test_LPRNet.py --test_img_dirs <test set> --pretrained_model weights/Final_LPRNet_model.pth
```

`results/` has one figure exported from the notebook.
