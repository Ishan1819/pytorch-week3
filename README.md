Overview

This repository contains PyTorch implementations of ResNet-18 (CIFAR-10) and a Transformer (toy translation) to understand core concepts from the original papers and experiment with key deep learning components.

Key Learnings

ResNet-18: Implemented residual blocks with identity & projection shortcuts; global average pooling reduced parameters; partial training reached 67% accuracy.

Transformer: Implemented multi-head attention & positional encodings; attention heatmaps showed input-output alignment.

Challenges & Resolutions

Mismatched residual dimensions → solved using 1×1 convolution projections.

Transformer divergence - fixed with learning rate warm-up and gradient clipping.

Hardware limitations - partial training and forward-pass testing only.

Practice Exercises

Forward pass on mini residual block using random tensors.

Implemented & validated scaled dot-product attention.

Debugged causal masks on toy sequence [1,2,3,4].


Setup Instructions

Clone the repository

git clone https://github.com/Ishan1819/pytorch-week3.git
cd pytorch-week3


Install dependencies

pip install -r requirements.txt


Run ResNet-18 training (partial due to hardware limits)

python code/train_resnet.py


Run Transformer toy translation

python code/train_transformer.py

Sources
ResNet

He, K., et al. (2015). Deep Residual Learning for Image Recognition. arXiv:1512.03385

PyTorch documentation (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear, torch.utils.data.DataLoader)

Grad-CAM: Selvaraju et al., 2016 arXiv:1610.02391

Transformer

Vaswani, A., et al. (2017). Attention Is All You Need. arXiv:1706.03762

Huang, H. Sequence-to-Sequence Modeling with nn.Transformer and TorchText (Tutorial)

NLTK documentation for BLEU score implementation (Papineni et al., 2002)

Notes

Due to hardware constraints, full-scale ResNet training is not performed; only partial training and functional testing are demonstrated.

Focus is on understanding architectures, forward passes, and attention mechanisms rather than achieving SOTA performance.
