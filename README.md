# Connecting the Digital and Physical World: Improving the Robustness of Adversarial Attacks (AAAI 2019)

This is the research code for the paper:

Steve T.K. Jan, Joseph Messou, Yen-Chen Lin, Jia-Bin Huang, Gang Wang "Connecting the Digital and Physical World: Improving the Robustness of Adversarial Attacks" 
In Proceedings of The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI), 2019

[Project page](http://people.cs.vt.edu/tekang/D2P/)

## Prerequisites
EOT.py is tested successfully in tensorflow 1.10 and python 3.6


## Usage

There are several steps to generate our physical attack

<img src="workflow.png"  width="500" />


1. Trained an image-to-image translation network. We used [implementations] (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
- [Pre-trained model](http://people.cs.vt.edu/tekang/D2P/latest_net_G.pth).
- [Data for the image-to-image network](http://people.cs.vt.edu/tekang/D2P/photos.zip)
- Here is the example script to use:
```bash
python test.py --dataroot ./datasets/test  --model test --checkpoints_dir ./checkpoints/  --name d2p/  --dataset_mode single --no_dropout  --norm batch
```
- ./datasets/test is the folder of images you want to transfer
- and put the model into the folder ./checkpoints/d2p/


2. Used EOT on simulated image.
Here are some parameters in the top of EOT.py that can be adjusted 

```bash
demo_steps=1000 # number of iterations
img_path='./simulated.JPEG' . # given an image
output_path = './demo.jpg' # output image
imagenet_json_path='./imagenet.json'
demo_epsilon = 40/255.0 # 
demo_lr = 0.5
demo_target = 21   ## adversarial class
img_class = 145  # original class
```
Generate noise: 
```bash
python EOT.py
```






### Citation


Please cite our paper if you find it useful for your research.

```
@inproceedings{aaai_d2p_2019,
  author = {Steve T.K. Jan and Joseph Messou and Yen-Chen Lin and Jia-Bin Huang and Gang Wang},
  booktitle = {The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)},
  title = {Connecting the Digital and Physical World: Improving the Robustness of
Adversarial Attacks},
  year = {2019}
}
Contact: Steve T.K. Jan (tekang at vt. edu)
```

## Acknowledgments

EOT implemenations is referenced in (https://www.anishathalye.com/2017/07/25/synthesizing-adversarial-examples/)
and thanks Yen-Chen Lin (http://yclin.me) for developing and testing


