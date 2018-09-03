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
Model	TF-Slim | File |	Checkpoint |	Top-1 Accuracy |	Top-5 Accuracy
Inception V1	Code	inception_v1_2016_08_28.tar.gz	69.8	89.6
Inception V2	Code	inception_v2_2016_08_28.tar.gz	73.9	91.8
Inception V3	Code	inception_v3_2016_08_28.tar.gz	78.0	93.9
Inception V4	Code	inception_v4_2016_09_09.tar.gz	80.2	95.2
Inception-ResNet-v2	Code	inception_resnet_v2_2016_08_30.tar.gz	80.4	95.3
ResNet V1 50	Code	resnet_v1_50_2016_08_28.tar.gz	75.2	92.2
ResNet V1 101	Code	resnet_v1_101_2016_08_28.tar.gz	76.4	92.9
ResNet V1 152	Code	resnet_v1_152_2016_08_28.tar.gz	76.8	93.2
ResNet V2 50^	Code	resnet_v2_50_2017_04_14.tar.gz	75.6	92.8
ResNet V2 101^	Code	resnet_v2_101_2017_04_14.tar.gz	77.0	93.7
ResNet V2 152^	Code	resnet_v2_152_2017_04_14.tar.gz	77.8	94.1
ResNet V2 200	Code	TBA	79.9*	95.2*
VGG 16	Code	vgg_16_2016_08_28.tar.gz	71.5	89.8
VGG 19	Code	vgg_19_2016_08_28.tar.gz	71.1	89.8
MobileNet_v1_1.0_224	Code	mobilenet_v1_1.0_224.tgz	70.9	89.9
MobileNet_v1_0.50_160	Code	mobilenet_v1_0.50_160.tgz	59.1	81.9
MobileNet_v1_0.25_128	Code	mobilenet_v1_0.25_128.tgz	41.5	66.3
MobileNet_v2_1.4_224^*	Code	mobilenet_v2_1.4_224.tgz	74.9	92.5
MobileNet_v2_1.0_224^*	Code	mobilenet_v2_1.0_224.tgz	71.9	91.0
NASNet-A_Mobile_224#	Code	nasnet-a_mobile_04_10_2017.tar.gz	74.0	91.6
NASNet-A_Large_331#	Code	nasnet-a_large_04_10_2017.tar.gz	82.7	96.2
PNASNet-5_Large_331	Code	pnasnet-5_large_2017_12_13.tar.gz	82.9	96.2
PNASNet-5_Mobile_224	Code	pnasnet-5_mobile_2017_12_13.tar.gz	74.2	91.9

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

# Face alignment  
(1)Look at Boundary: A Boundary-Aware Face Alignment Algorithm
https://github.com/wywu/LAB  

# Detection  
(1)maskrcnn https://arxiv.org/abs/1703.06870  
mx:https://github.com/TuSimple/mx-maskrcnn  
tf:https://github.com/CharlesShang/FastMaskRCNN  
keras+tf:https://github.com/matterport/Mask_RCNN  
(2)FCIS https://arxiv.org/abs/1611.07709  
https://github.com/msracver/FCIS  

# Organization  
microsoft:https://github.com/Microsoft  
facebook:https://github.com/facebookresearch  
google:https://github.com/google  
opencv:https://github.com/opencv  
deepinsight:https://github.com/deepinsight  
cmusatyalab:https://github.com/cmusatyalab  
nvidia:https://github.com/NVIDIA  
chainer:https://github.com/chainer  
pytorch:https://github.com/pytorch  
baidu:https://github.com/baidu-research  
pydicom:https://github.com/pydicom  

# Article  
cvpr,iccv:http://openaccess.thecvf.com/menu.py  
eccv:https://dblp.uni-trier.de/db/conf/eccv/index.html  
