"""
CUDA Enablement and Diagnostic Script
This script checks CUDA availability and configures PyTorch to use GPU
"""
import os
import sys

def check_cuda_support():
    """Check if CUDA is available and display GPU information"""
    try:
        import torch
        
        print("=" * 60)
        print("CUDA DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # PyTorch version
        print(f"\n[PyTorch] Version: {torch.__version__}")
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"[CUDA] Available: {cuda_available}")
        
        if cuda_available:
            # CUDA version
            print(f"[CUDA] Version: {torch.version.cuda}")
            
            # GPU count
            gpu_count = torch.cuda.device_count()
            print(f"[GPU] Count: {gpu_count}")
            
            # GPU details
            for i in range(gpu_count):
                print(f"\n--- GPU {i} Details ---")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
                print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # Current device
            current_device = torch.cuda.current_device()
            print(f"\n[SUCCESS] Current CUDA Device: {current_device} ({torch.cuda.get_device_name(current_device)})")
            
            # Test tensor creation on GPU
            print("\n[TEST] Testing GPU tensor creation...")
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"  Test tensor device: {test_tensor.device}")
            print(f"  [SUCCESS] GPU tensor creation successful!")
            
        else:
            print("\n[ERROR] CUDA is NOT available!")
            print("\nPossible reasons:")
            print("  1. PyTorch was installed without CUDA support (CPU-only version)")
            print("  2. NVIDIA GPU drivers are not installed")
            print("  3. No compatible NVIDIA GPU detected")
            
            print("\n[TIP] To install PyTorch with CUDA support, run:")
            print("  pip uninstall torch torchvision torchaudio -y")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
        print("\n" + "=" * 60)
        return cuda_available
        
    except ImportError:
        print("[ERROR] PyTorch is not installed!")
        print("Please install PyTorch first: pip install torch")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False

def enable_cuda_optimizations():
    """Enable CUDA optimizations and set environment variables"""
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA not available. Skipping optimizations.")
        return
    
    print("\n[OPTIMIZING] Enabling CUDA Optimizations...")
    
    # Set GPU as default device
    torch.cuda.set_device(0)
    print(f"  [OK] Set default GPU device to: cuda:0")
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    print(f"  [OK] Enabled cuDNN benchmark mode")
    
    # Enable cuDNN deterministic mode (optional, for reproducibility)
    # torch.backends.cudnn.deterministic = True
    # print(f"  [OK] Enabled cuDNN deterministic mode")
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print(f"  [OK] Set CUDA memory allocation config")
    
    print("\n[SUCCESS] CUDA optimizations enabled!")

def test_gpu_computation():
    """Test a simple GPU computation"""
    if not torch.cuda.is_available():
        return
    
    print("\n[TEST] Testing GPU Computation...")
    
    try:
        # Create tensors on GPU
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        # Perform computation
        import time
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end = time.time()
        
        print(f"  Matrix multiplication (1000x1000) completed in {(end-start)*1000:.2f}ms")
        print(f"  Result device: {c.device}")
        print(f"  [SUCCESS] GPU computation successful!")
        
    except Exception as e:
        print(f"  [ERROR] GPU computation failed: {e}")

if __name__ == "__main__":
    print("\n>>> CUDA Enablement Script <<<\n")
    
    # Import torch (needed for other functions)
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch not installed. Please install it first.")
        sys.exit(1)
    
    # Check CUDA support
    cuda_available = check_cuda_support()
    
    if cuda_available:
        # Enable optimizations
        enable_cuda_optimizations()
        
        # Test GPU computation
        test_gpu_computation()
        
        print("\n[SUCCESS] CUDA is ready to use!")
        print("\n[TIP] Your code should now automatically use GPU when you specify:")
        print("   device = 'cuda' if torch.cuda.is_available() else 'cpu'")
        print("   model.to(device)")
    else:
        print("\n[ERROR] CUDA setup incomplete. Please resolve the issues above.")
        sys.exit(1)

