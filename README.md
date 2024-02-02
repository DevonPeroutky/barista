# It's a really good repo

### Random
You may need to add this to your bashrc to prevent CUDA dependency issues

```
export $LD_LIBRARY_PATH="/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:$PATH"
```

Pytorch binaries come with their own CUDA dependencies built-in. GCP VMs will also have the CUDA toolkit installed and this version mismatch can cause issues. So remove
the path to the installed CUDA dependencies (or remove them) and pytorch will fall back to it's bundled dependencies.

[Source](https://discuss.pytorch.org/t/could-not-load-library-libcudnn-cnn-train-so-8-in-new-version/190818)
