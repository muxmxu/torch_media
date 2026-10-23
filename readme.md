# TorchMedia

🚀 **TorchMedia** is a **header-only** C++ library designed for seamless media processing. Built on **native LibTorch APIs**, it brings the power of **TorchAudio** and **TorchVision** into a lightweight, plug-and-play package. And yes, **vision processing is coming soon!** Stay tuned! 👀

## Why TorchMedia? 🤔

- **Header-only & Hassle-free** ✨: No complex builds—just include and start coding.
- **Powered by LibTorch** 🔥: Seamless integration with PyTorch's C++ backend.
- **Audio & Vision Processing** 🎵📷: Handle audio now, and soon images too!
- **Familiar Torch-style API** 🏗️: If you use `torchaudio` and `torchvision`, you'll feel right at home.
- **Multi-platform & Multi-backend** ⚡: Run effortlessly on **CPU, CUDA, MPS**, or any backend supported by PyTorch.

## Installation 🎯

No installation needed—just clone and include the headers:

```sh
# Clone the repository
git clone https://github.com/your-repo/torch_media.git
```

## Quick Start 🎬

Load an audio file and compute its spectrogram in just a few lines:

```cpp
#include <libtorchmedia.hpp>

int main() {
    using io = torchmedia::audio::io;
    using functional = torchmedia::audio::functional;
    
    // Load an audio file 
    auto my_audio = io::load_audio("path/to/audio.wav").data; 
    
    // Compute the spectrogram
    auto spectrogram = functional::spectrogram(
        my_audio, functional::spectrogram_options_t{}.windows(400).n_fft(200)
    );
    
    // Print the result
    std::print("{}", torchmedia::basic::to_string(spectrogram));
    
    return 0;
}
```

## Dependencies 🛠️

TorchMedia is lightweight but relies on a few essentials:

- **LibTorch** (any version) – PyTorch's C++ frontend.
- **SOX** (any version) – For audio processing.

## License 📜

TorchMedia is open-source under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing 💡

We welcome contributions! Found a bug or have a feature request? Open an issue or submit a pull request. Let’s build something amazing together. 🚀

## Get in Touch 🎤

Have questions or feedback? Drop by the GitHub Issues—we’d love to hear from you!

