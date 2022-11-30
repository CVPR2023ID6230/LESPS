# Mapping Degeneration Meets Label Evolution: Learning Infrared Small Target Detection with Single Point Supervision

Pytorch implementation of our Label Evolution with Single Point Supervision (LESPS).<br><br>

## Overview

### The Mapping Degeneration Phenomenon
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/MD1.jpg" width="550"/><br>
Figure 1. Illustrations of mapping degeneration under point supervision. CNNs always tend to segment a cluster of pixels near the targets with low confidence at the early stage, and then gradually learn to predict groundtruth point labels with high confidence.

<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/MD2.jpg" width="550"/><br>
Figure 2. Quantitative and qualitative illustrations of mapping
degeneration in CNNs.


### The Label Evolution Framework
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/LESPS.jpg" width="550"/><br>
Figure 3. Illustrations of Label Evolution with Single Point
Supervision (LESPS). During training, intermediate predictions of CNNs are used to progressively expand point labels to mask labels. Black arrows represent each round of label updates.

## Requirements
- Python 3
- pytorch (1.2.0), torchvision (0.4.0) or higher
- numpy, PIL

## Datasets
* [The IRSTD-1K download dir](https://github.com/RuiZhang97/ISNet) [[ISNet]](https://ieeexplore.ieee.org/document/9880295)
* [The NUDT-SIRST download dir](https://github.com/YeRen123455/Infrared-Small-Target-Detection) [[NUDT]](https://ieeexplore.ieee.org/abstract/document/9864119)
* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)
* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) [[MDvsFA]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)

## Test
```bash
python test.py --model DNAnet_centroid
```
--model (Optional: DNAnet_full, DNAnet_centroid, DNAnet_coarse, ACM_full, ACM__centroid, ACM__coarse, ALCnet_full, ALCnet_centroid, ALCnet_coarse)

## Results

### Analyses of Mapping Degeneration
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/MD_abl.jpg" width="800"/><br>
Figure 4. IoU and visualize results of mapping degeneration with respect to different characteristics of targets (i.e.,(a) intensity, (b) size, (c) shape, and (d) local background clutter) and point labels (i.e.,(e) numbers and (f) locations). We visualize the zoom-in target regions of input images with GT point labels (i.e., red dots in images) and corresponding CNN predictions (in the epoch reaching maximum IoU).

### Analyses of the Label Evolution Framework

#### Effectiveness
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/table0.jpg" width="500"  /><br>
Table 1. Average IoU (×10e2), Pd (×10e2) and Fa(×10e6) values on NUAA-SIRST, NUDT-SIRST and IRSTD-1K achieved by DNAnet with (w/) and without (w/o) LESPS under centroid, coarse point supervision together with full supervision.

<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/visual1.jpg" width="600"/><br>
Figure 6. Visualizations of regressed labels during training and
network predictions during inference with centroid and coarse
point supervision.

#### Parameters
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/LESPS_abl.jpg" width="800"/><br>
Figure 5. PA (P) and IoU (I) results of LESPS with respect to (a) initial evolution loss Tloss, (b) Tb and (c) k of evolution threshold, and (d) evolution frequency f.

### Quantitative Results

#### Comparison to SISRT detection methods.

<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/tabel1.jpg" width="550"/><br>
Table 2.IoU (×10e2), Pd (×10e2) and Fa(×10e6) values of different methods achieved on NUAA-SIRST, NUDT-SIRST and IRSTD-1K. “CNN Full”, “CNN Centroid”, and “CNN Coarse” represent CNN-based methods under full supervision, centroid and coarse point supervision. “+” represents CNN-based methods equipped with LESPS.

<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/visual2.jpg" width="800"/><br>
Figure 7. Visual detection results of different methods achieved
on NUAA-SIRST, NUDT-SIRST and IRSTD-1K. Correctly detected targets and false alarms are highlighted by red and orange circles, respectively.

#### Comparison to LCM-based pseudo labels. 
<img src="https://raw.github.com/CVPR2023ID6230/LESPS/master/Figs/tabel2.jpg" width="550"/><br>
Table 3. Average IoU (×10e2), Pd (×10e2) and Fa(×10e6) values on NUAA-SIRST, NUDT-SIRST and IRSTD-1K
of DNAnet trained with pseudo labels generated by different
LCM-based methods and LESPS under centroid and coarse point
supervision.

