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

# Detection  
my blog:https://blog.csdn.net/u013066730/article/details/82460392  
(1)maskrcnn https://arxiv.org/abs/1703.06870  
mx:https://github.com/TuSimple/mx-maskrcnn  
tf:https://github.com/CharlesShang/FastMaskRCNN  
keras+tf:https://github.com/matterport/Mask_RCNN  
(2)FCIS https://arxiv.org/abs/1611.07709  
https://github.com/msracver/FCIS  
(3)SSD http://arxiv.org/abs/1512.02325
caffe:https://github.com/weiliu89/caffe/tree/ssd  

# Classification
(1)PCANet https://arxiv.org/pdf/1404.3606.pdf  
chainer:https://github.com/IshitaTakeshi/PCANet  
scalar c++:https://github.com/Ldpe2G/PCANet  

# Segmentation
(1)Linknet:https://arxiv.org/abs/1707.03718  
https://codeac29.github.io/projects/linknet/  
lua:https://github.com/mjiansun/LinkNet  

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

# Article  
cvpr,iccv:http://openaccess.thecvf.com/menu.py  
eccv:https://dblp.uni-trier.de/db/conf/eccv/index.html  

# Live video
https://github.com/stlndm/linke
https://github.com/ChinaArJun/Tencent-NOW  

# Course
1）Stanford:https://www.coursera.org/learn/machine-learning  
code:https://github.com/yoyoyohamapi/mit-ml
