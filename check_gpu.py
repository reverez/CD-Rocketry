"""Verify that the project can see PyTorch and an NVIDIA CUDA GPU."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.rand((2048, 2048), device=device)
        y = x @ x
        torch.cuda.synchronize()
        print(f"CUDA matmul OK: {float(y[0, 0]):.4f}")
    else:
        print("No CUDA GPU visible to PyTorch. Check the NVIDIA driver and CUDA PyTorch wheel.")


if __name__ == "__main__":
    main()
