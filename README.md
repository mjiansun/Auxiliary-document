# Tensorflow
## Doc

https://github.com/tensorflow/models/tree/master/research
TensorFlow Research Models

This folder contains machine learning models implemented by researchers in TensorFlow. The models are maintained by their respective authors. To propose a model for inclusion, please submit a pull request.

Currently, the models are compatible with TensorFlow 1.0 or later. If you are running TensorFlow 0.12 or earlier, please upgrade your installation.

## Models

adversarial_crypto: protecting communications with adversarial neural cryptography.  
adversarial_text: semi-supervised sequence learning with adversarial training.  
attention_ocr: a model for real-world image text extraction.  
audioset: Models and supporting code for use with AudioSet.  
autoencoder: various autoencoders.  
brain_coder: Program synthesis with reinforcement learning.  
cognitive_mapping_and_planning: implementation of a spatial memory based mapping and planning architecture for visual navigation.  
compression: compressing and decompressing images using a pre-trained Residual GRU network.  
deeplab: deep labelling for semantic image segmentation.  
delf: deep local features for image matching and retrieval.  
differential_privacy: privacy-preserving student models from multiple teachers.  
domain_adaptation: domain separation networks.  
gan: generative adversarial networks.  
im2txt: image-to-text neural network for image captioning.  
inception: deep convolutional networks for computer vision.  
learning_to_remember_rare_events: a large-scale life-long memory module for use in deep learning.  
lexnet_nc: a distributed model for noun compound relationship classification.  
lfads: sequential variational autoencoder for analyzing neuroscience data.  
lm_1b: language modeling on the one billion word benchmark.  
maskgan: text generation with GANs.  
namignizer: recognize and generate names.  
neural_gpu: highly parallel neural computer.  
neural_programmer: neural network augmented with logic and mathematic operations.  
next_frame_prediction: probabilistic future frame synthesis via cross convolutional networks.  
object_detection: localizing and identifying multiple objects in a single image.  
pcl_rl: code for several reinforcement learning algorithms, including Path Consistency Learning.  
ptn: perspective transformer nets for 3D object reconstruction.  
qa_kg: module networks for question answering on knowledge graphs.  
real_nvp: density estimation using real-valued non-volume preserving (real NVP) transformations.  
rebar: low-variance, unbiased gradient estimates for discrete latent variable models.  
resnet: deep and wide residual networks.  
skip_thoughts: recurrent neural network sentence-to-vector encoder.  
slim: image classification models in TF-Slim.  
street: identify the name of a street (in France) from an image using a Deep RNN.  
swivel: the Swivel algorithm for generating word embeddings.  
syntaxnet: neural models of natural language syntax.  
tcn: Self-supervised representation learning from multi-view video.  
textsum: sequence-to-sequence with attention model for text summarization.  
transformer: spatial transformer network, which allows the spatial manipulation of data within the network.  
video_prediction: predicting future video frames with neural advection. 

# Slim  
https://github.com/tensorflow/models/tree/master/research/slim  
![Slim model](https://github.com/mjiansun/Auxiliary-document/blob/master/slim_model.png "Slim model")
## Nasnet  
https://blog.csdn.net/qq_36356761/article/details/79521694  

# Mxnet(啥都有)
https://github.com/dmlc/gluon-cv  

# Pytorch  
https://github.com/pytorch/examples  
https://github.com/pytorch/vision (this is torchvision source code)  
(1)Image classification (MNIST) using Convnets  
(2)Word level Language Modeling using LSTM RNNs  
(3)Training Imagenet Classifiers with Residual Networks  
(4)Generative Adversarial Networks (DCGAN)  
(5)Variational Auto-Encoders  
(6)Superresolution using an efficient sub-pixel convolutional neural network  
(7)Hogwild training of shared ConvNets across multiple processes on MNIST  
(8)Training a CartPole to balance in OpenAI Gym with actor-critic  
(9)Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext  
(10)Time sequence prediction - use an LSTM to learn Sine waves  
(11)Implement the Neural Style Transfer algorithm on images  
(12)Several examples illustrating the C++ Frontend  

# Transform style
(1)Perceptual Losses for Real-Time Style Transfer and Super-Resolution(fast-neural-style)  
https://www.jianshu.com/p/b728752a70e9  
lua:https://github.com/jcjohnson/fast-neural-style  
tensorflow:https://github.com/hzy46/fast-neural-style-tensorflow  
pytorch:https://github.com/abhiskk/fast-neural-style  
chainer:https://github.com/yusuketomoto/chainer-fast-neuralstyle   
(2)Texture Networks: Feed-forward Synthesis of Textures and Stylized Images(texture_nets)  
https://www.jianshu.com/p/1187049ae1ad   
tensorflow:https://github.com/tgyg-jegli/tf_texture_net  
lua:https://github.com/DmitryUlyanov/texture_nets  
(3)Instance Normalization: The Missing Ingredient for Fast Stylization(IN)  
https://www.jianshu.com/p/d77b6273b990  
lua:https://github.com/DmitryUlyanov/texture_nets  
(4)Fast Neural Style Transfer with Arbitrary Style using AdaIN Layer(AdaIN)  
lua:https://github.com/xunhuang1995/AdaIN-style
<br/>pytorch:https://github.com/naoto0804/pytorch-AdaIN</br>
tensorflow:https://github.com/elleryqueenhomels/arbitrary_style_transfer  
(5)Controlling Perceptual Factors in Neural Style Transfer  
lua:https://github.com/leongatys/NeuralImageSynthesis  
(6)Deep Photo Style Transfer  
https://blog.csdn.net/cicibabe/article/details/70868746  
lua:https://github.com/luanfujun/deep-photo-styletransfer  
tensorflow:https://github.com/NVIDIA/FastPhotoStyle  
tensorflow:https://github.com/LouieYang/deep-photo-styletransfer-tf  
pytorch:https://github.com/NVIDIA/FastPhotoStyle  
(7)Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks(cyclegan)  
torch:https://github.com/junyanz/CycleGAN  
pytorch:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  

# Defuzzification  
(1)DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks  
https://arxiv.org/pdf/1711.07064.pdf  
keras:https://github.com/RaphaelMeudec/deblur-gan  

# Simultaneous interpretation
(1)STACL: Simultaneous Translation with Integrated Anticipation and Controllable Latency  
https://arxiv.org/pdf/1810.08398.pdf  

# Face alignment  
(1)Look at Boundary: A Boundary-Aware Face Alignment Algorithm  
https://github.com/wywu/LAB  

# Backbone network  
(1)Mobile Net v1(MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications)  
caffe:https://github.com/shicai/MobileNet-Caffe  
(2)Mobile Net v2(MobileNetV2: Inverted Residuals and Linear Bottlenecks)  
caffe:https://github.com/shicai/MobileNet-Caffe  
(3)Resnet(Deep Residual Learning for Image Recognition)  
pytorch:https://github.com/Cadene/pretrained-models.pytorch  
(4)VGG(Very Deep Convolutional Networks for Large-Scale Image Recognition)  
(5)Google Net  
https://blog.csdn.net/u011974639/article/details/76460849#googlenet  
Inception V1:Going deeper with convolutions  
Inception V2:Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
Inception V3:Rethinking the Inception Architecture for Computer Vision  
Inception V4:Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning  

# Detection  
https://github.com/hoya012/deep_learning_object_detection  
my blog:https://blog.csdn.net/u013066730/article/details/82460392  
https://github.com/tensorflow/models/tree/master/research/object_detection  
https://paperswithcode.com/sota/object-detection-on-coco(This is a good web.)  
(1)maskrcnn https://arxiv.org/abs/1703.06870  
mx:https://github.com/TuSimple/mx-maskrcnn  
tf:https://github.com/CharlesShang/FastMaskRCNN  
keras+tf:https://github.com/matterport/Mask_RCNN  
pytorch:https://github.com/facebookresearch/maskrcnn-benchmark  
caffe2:https://github.com/facebookresearch/Detectron  
(2)FCIS https://arxiv.org/abs/1611.07709  
https://github.com/msracver/FCIS  
(3)SSD http://arxiv.org/abs/1512.02325  
caffe:https://github.com/weiliu89/caffe/tree/ssd  
(4)M2Det https://qijiezhao.github.io/imgs/m2det.pdf  
pytorch:https://github.com/qijiezhao/M2Det  
(5)efficientdet  
keras:https://github.com/xuannianz/EfficientDet  
(6)yolov3  
keras:https://github.com/OlafenwaMoses/ImageAI#detection  
(7)efficientdet  
tensorflow:https://github.com/google/automl/tree/master/efficientdet  

# Classification  
https://github.com/Cadene/pretrained-models.pytorch  
https://github.com/keras-team/keras-applications  
https://github.com/qubvel/classification_models  
(1)PCANet https://arxiv.org/pdf/1404.3606.pdf  
chainer:https://github.com/IshitaTakeshi/PCANet  
scalar c++:https://github.com/Ldpe2G/PCANet  
(2)CBAM&BAM:https://github.com/Jongchan/attention-module  
(3)efficientnet  
keras:https://github.com/qubvel/efficientnet  


# Segmentation  
https://blog.playment.io/semantic-segmentation-models-autonomous-vehicles/  
https://github.com/mrgloom/awesome-semantic-segmentation  
https://mp.weixin.qq.com/s/w7pYxm52QbcFPRBe12iMdA and https://arxiv.org/abs/2001.05566  
好用：https://github.com/qubvel/segmentation_models.pytorch and https://github.com/qubvel/segmentation_models  
(1)Linknet:https://arxiv.org/abs/1707.03718  
https://codeac29.github.io/projects/linknet/  
lua:https://github.com/mjiansun/LinkNet  
pytorch:https://github.com/e-lab/pytorch-linknet  

# Retrieval  
https://github.com/willard-yuan/awesome-cbir-papers  

# GAN  
tensorflow:https://github.com/hwalsuklee/tensorflow-generative-model-collections  
pytorch:https://github.com/znxlwm/pytorch-generative-model-collections  
article:https://github.com/zhangqianhui/AdversarialNetsPapers  
(1)stargan   
pytorch:https://github.com/yunjey/StarGAN  
(2)cyclegan  
pytorch:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  
(3)wgan  
pytorch:https://github.com/caogang/wgan-gp  
(4)staingan  
article&code:https://xtarx.github.io/StainGAN/  
## GAN FOR MEDICAL  
https://github.com/xinario/awesome-gan-for-medical-imaging  

# Instance Segmentation  
RDSNet:https://arxiv.org/abs/1912.05070  
https://github.com/wangsr126/RDSNet  
YOLACT:https://arxiv.org/abs/1904.02689  
https://github.com/dbolya/yolact  
YOLACT++:https://arxiv.org/abs/1912.06218  
https://github.com/dbolya/yolact  
CenterMask:https://arxiv.org/abs/1911.06667  
https://github.com/youngwanLEE/CenterMask  
maskrcnn:https://arxiv.org/abs/1703.06870  
https://github.com/matterport/Mask_RCNN  

# Medical image registration  
https://github.com/DeepRegNet/DeepReg  

# Imbalanced learning  
https://github.com/ZhiningLiu1998/awesome-imbalanced-learning  

# Model Compression  
以下参考：https://blog.csdn.net/nature553863/article/details/81083955  
模型压缩算法能够有效降低参数冗余，从而减少存储占用、通信带宽和计算复杂度，有助于深度学习的应用部署，具体可划分为如下几种方法：  
**(1)线性或非线性量化：1/2bits, int8 和 fp16等**  
模型量化是指权重或激活输出可以被聚类到一些离散、低精度（reduced precision）的数值点上，通常依赖于特定算法库或硬件平台的支持：  
+ 二值化网络：XNORnet [https://arxiv.org/abs/1603.05279,  Github: https://github.com/ayush29feb/Sketch-A-XNORNet, Github: https://github.com/jiecaoyu/XNOR-Net-PyTorch], ABCnet with Multiple Binary Bases [https://arxiv.org/abs/1711.11294,  Github: https://github.com/layog/Accurate-Binary-Convolution-Network], Bin-net with High-Order Residual Quantization [https://arxiv.org/abs/1708.08687], Bi-Real Net [https://arxiv.org/abs/1808.00278, Github: https://github.com/liuzechun/Bi-Real-net]；  
+ 三值化网络：Ternary weight networks [https://arxiv.org/abs/1605.04711], Trained Ternary Quantization [https://arxiv.org/abs/1612.01064,  Github: https://github.com/czhu95/ternarynet]；  
+ W1-A8 或 W2-A8量化： Learning Symmetric Quantization [http://phwl.org/papers/syq_cvpr18.pdf,  Github: https://github.com/julianfaraone/SYQ]；  
+ INT8量化：TensorFlow-lite [https://arxiv.org/abs/1712.05877], TensorRT [http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf], Quantization Interval Learning [https://blog.csdn.net/nature553863/article/details/96857133]；  
+ INT4量化：NVIDIA Iterative Online Calibration [https://blog.csdn.net/nature553863/article/details/104080434], LSQ [https://blog.csdn.net/nature553863/article/details/104275477];  
+ 其他（非线性）：Intel INQ [https://arxiv.org/abs/1702.03044], log-net, CNNPack [https://papers.nips.cc/paper/6390-cnnpack-packing-convolutional-neural-networks-in-the-frequency-domain] 等；  
+ Post-training量化策略：针对预训练模型，通过适当调整kernel参数分布、或补偿量化误差，可有效提升量化效果；  
+ 关于量化的比较系统性的论述，参考论文：Quantizing deep convolutional networks for efficient inference: A whitepaper；  


**(2)结构或非结构剪枝：deep compression, channel pruning 和 network slimming等；**  
**非结构剪枝**：通常是连接级、细粒度的剪枝方法，精度相对较高，但依赖于特定算法库或硬件平台的支持，如Deep Compression [https://arxiv.org/abs/1510.00149], Sparse-Winograd [https://arxiv.org/abs/1802.06367,  https://ai.intel.com/winograd-2/, Github: https://github.com/xingyul/Sparse-Winograd-CNN] 算法等；
**结构剪枝**：是filter级或layer级、粗粒度的剪枝方法，精度相对较低，但剪枝策略更为有效，不需要特定算法库或硬件平台的支持，能够直接在成熟深度学习框架上运行:  
+ 如局部方式的、通过layer by layer方式的、最小化输出FM重建误差的Channel Pruning [https://arxiv.org/abs/1707.06168,  Github: https://github.com/yihui-he/channel-pruning], ThiNet [https://arxiv.org/abs/1707.06342], Discrimination-aware Channel Pruning [https://arxiv.org/abs/1810.11809, Github: https://github.com/Tencent/PocketFlow]；  
+ 全局方式的、通过训练期间对BN层Gamma系数施加L1正则约束的Network Slimming [https://arxiv.org/abs/1708.06519,  Github: https://github.com/foolwood/pytorch-slimming]；  
+ 全局方式的、按Taylor准则对Filter作重要性排序的Neuron Pruning [https://arxiv.org/abs/1611.06440,  Github: https://github.com/jacobgil/pytorch-pruning]；  
+ 全局方式的、可动态重新更新pruned filters参数的剪枝方法 [http://xuanyidong.com/publication/ijcai-2018-sfp/]；  
+ 基于GAN思想的GAL方法 [https://blog.csdn.net/nature553863/article/details/97631176]，可裁剪包括Channel, Branch或Block等在内的异质结构；  
+ 借助Geometric Median确定卷积滤波器冗余性的剪枝策略 [https://blog.csdn.net/nature553863/article/details/97760040]；  
+ 基于Reinforcement Learning (RL)，实现每一层剪枝率的连续、精细控制，并可结合资源约束完成自动模型压缩 [https://github.com/mit-han-lab/amc];  


**(3)网络结构搜索 (NAS: Network Architecture Search)：DARTS, DetNAS、NAS-FCOS、Proxyless NAS和NetAdapt等；**  


**(4)其他：权重矩阵的低秩分解，知识蒸馏与网络结构简化（squeeze-net, mobile-net, shuffle-net）等；**  
知识蒸馏介绍：https://blog.csdn.net/u013066730/article/details/111573882  
百度的知识蒸馏SSLD：https://github.com/PaddlePaddle/PaddleClas/blob/master/docs/zh_CN/advanced_tutorials/distillation/distillation.md

# TOOL  
Pytorch2caffe:https://github.com/xxradon/PytorchToCaffe  

# Organization  
chainer:https://github.com/chainer  
pytorch:https://github.com/pytorch  
pydicom:https://github.com/pydicom  
mxnet:https://github.com/dmlc  
baidu:https://github.com/baidu-research  
tencent：https://github.com/Tencent  
tencent Ai Lab：https://ai.tencent.com/ailab/index.html
microsoft:https://github.com/Microsoft  
facebook:https://github.com/facebookresearch  
google:https://github.com/google  
opencv:https://github.com/opencv  
deepinsight:https://github.com/deepinsight  
cmusatyalab:https://github.com/cmusatyalab  
nvidia:https://github.com/NVIDIA  
Purdue University:https://github.com/e-lab  

# Article  
cvpr,iccv:http://openaccess.thecvf.com/menu.py  
eccv:https://dblp.uni-trier.de/db/conf/eccv/index.html  

# Live video
https://github.com/stlndm/linke
https://github.com/ChinaArJun/Tencent-NOW  

# Course
1）Stanford:https://www.coursera.org/learn/machine-learning  
code:https://github.com/yoyoyohamapi/mit-ml
