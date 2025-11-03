# Requirements

To run the ILP part of HAWQ-V3, it is mandatory to have the PULP package and the GLPK_CMD solver. To insatll the same:
```console
pip install pulp
apt-get install -y coinor-cbc glpk-utils coinor-clp
```

# Preface
For the benchmarking, I have used the VGG-16 model for CIFAR-10 dataset from [https://github.com/chenyaofo/pytorch-cifar-models].

# ILP
To run just the ILP in order to get the bit-mapping for the VGG-16 model, just simply run the `ILP.py` file. There are two arguments that can be passed:
1. `--modified_hawq`: If true, it gives the modified bit mapping with 2-8 bit precision while it gives the original \[4,8\] bit mappping otherwise. Default: true
2. `--bops_limit_factor`: Gives the percentage ratio between 4-bit bops and 8-bit bops, i.e., bops\_limit = bops\_4bit + (bops\_8bit - bops\_4bit)*bops\_factor. Default: 0.5

# Main Inference
To run the full flow which includes finding the bit mapping as well as final accuracy, run the `main.py` file. The following arguments are available:
1. `--uniform`: If true, it does single-precision quantisation for the entire network. Default: False
2. `--uniform_bits`: If the above is true, the bit-precision of weights. Default: 8
3. `--modified_hawq`: Same as above
4. `--bops_limit_factor`: Same as above
5. `--bn_fold`: Wether to fold BN layers into the previous conv layers. Default: True
6. `--fine_tune`: Enable fine-tuning after quantization. Deault: True
7. `--lr`: Learning rate for fine-tuning. Deault: 1e-4
8. `--num_epochs`: Number of fine-tuning epochs. Deault: 10
9. `--save_file`: Path to save the best 
