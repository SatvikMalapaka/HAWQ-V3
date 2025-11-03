import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from utils import *
from ILP import ilp_main_bops


def main(args):
    # Transformations for training and test sets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)

    print("\nBaseline Accuracy")
    evaluate(model, testloader, device)
    print("\n")

    if args.uniform:
        model_q = quantize_model(model, w_bits=args.uniform_bits, a_bits=8)
        print(f"\nUniform {args.uniform_bits}-bit quantised model accuracy:")
        evaluate(model_q, testloader, device)

    else:
        if not args.modified_hawq:
            w_bits_dict, bops = ilp_main_bops(model, modified_hawq=False, bops_limit_factor=args.bops_limit_factor)
            model = model.to(device)
            model_q = quantize_model_layerwise(model, w_bits_dict)
            print("\nAccuracy when using Native Hawq implementation:")
            evaluate(model_q, testloader, device)
            print("\nNumber of bit operations:", bops)

        else:
            w_bits_dict, bops = ilp_main_bops(model, modified_hawq=True, bops_limit_factor=args.bops_limit_factor)
            model = model.to(device)

            if args.bn_fold:
                model.eval()
                model = fuse_module_conv_bn(model)
                model_q = quantize_model_layerwise_LearnableQuantizer(model, w_bits_dict)
                recalibrate_bn(model_q, trainloader, device, num_batches=10)
                print("\nAccuracy without fine-tuning:")
                evaluate(model_q, testloader, device)

                if args.fine_tune:
                    fine_tune(model_q, trainloader, testloader, args.num_epochs, args.lr, device, args.save_file)

                print("\nNumber of bit operations:", bops)
            else:
                model_q = quantize_model_layerwise(model, w_bits_dict)
                print("\nAccuracy when using Native Hawq implementation:")
                evaluate(model_q, testloader, device)
                print("\nNumber of bit operations:", bops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization and HAWQ experiment runner")

    parser.add_argument("--uniform", type=bool, default=False, help="Use uniform quantization")
    parser.add_argument("--uniform_bits", type=int, default=8, help="Bitwidth for uniform quantization")
    parser.add_argument("--modified_hawq", type=bool, default=True, help="Use modified HAWQ")
    parser.add_argument("--bops_limit_factor", type=float, default=0.5, help="BOPS limit factor for HAWQ")
    parser.add_argument("--bn_fold", type=bool, default=True, help="Whether to fold BN before quantization")
    parser.add_argument("--save_file", type=str, default="best_finetuned_quantized.pth", help="Path to save fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--fine_tune", action="store_true", help="Enable fine-tuning after quantization")

    args = parser.parse_args()
    main(args)
