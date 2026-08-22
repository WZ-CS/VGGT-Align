<h1 align="center">VGGT-Align: Bridging Local Reconstruction and Global Consistency for Long-Sequence 3D Reconstruction</h1>

<p align="center">
  <strong><a href="https://arxiv.org/abs/2608.15260">Paper</a></strong>
</p>

Official implementation of **VGGT-Align**, a framework for improving global geometric consistency in long-sequence 3D reconstruction.

**Abstract:** Maintaining global geometric consistency is a central challenge in long-sequence 3D reconstruction, with scale drift being the most critical failure mode. In chunk-based inference pipelines, the scale degree of freedom in sequential Sim(3) alignment is left unconstrained, causing estimation errors to compound multiplicatively and distort global trajectories and point cloud geometry. We present a scale-consistency enhancement framework built on a key insight: in structured environments such as driving scenes, geometric quantities arising from environmental regularity remain inherently invariant across temporal segments, and discrepancies in their per-chunk measurements directly expose inter-chunk scale drift. We propose Scene Geometric Invariant Anchoring (SGIA), which extracts dominant geometric invariants from each chunk's predicted point cloud via coarse-to-fine robust estimation and exploits their cross-chunk consistency to establish scale constraints independent of point cloud registration, explicitly degenerating 7-DoF Sim(3) alignment into a 6-DoF rigid-body transformation and preventing chain-wise scale-error propagation at its source. We further introduce a lightweight test-time adaptation strategy that fine-tunes only normalization-layer parameters through multi-objective self-supervision, progressively improving intra-chunk predictions along the sequence. Both modules are plug-and-play and require no offline retraining. Experiments on multiple long-sequence benchmarks demonstrate state-of-the-art performance, reducing absolute trajectory error by up to 32% while improving trajectory stability and reconstruction quality.

![Overview](./assets/overview.png)

## News

- **July 10, 2026:** Accepted by ACM MM 2026.

## Setup and Installation

### 1. Create the environment

```bash
conda create -n vggt-align python=3.10.18
conda activate vggt-align
```

Install PyTorch:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download pretrained weights

The script downloads the required VGGT, Pi3, and MapAnything weights:

```bash
bash ./scripts/download_weights.sh
```

The following compilation steps are optional. Skip them to run VGGT-Align entirely in Python.

### 3. Compile the loop-closure correction module (optional)

VGGT-Align includes a Python Sim(3) solver, but the C++ implementation is faster and more stable:

```bash
python setup.py install
```

### 4. Compile the DBoW loop-closure detection module (optional)

This module enables CPU-only visual place recognition.

<details>
  <summary><strong>Installation instructions</strong></summary>

Install the OpenCV C++ API:

```bash
sudo apt-get install -y libopencv-dev
```

Build and install DBoW2:

```bash
cd DBoW2
mkdir -p build
cd build
cmake ..
make
sudo make install
cd ../..
```

Install the image-retrieval module:

```bash
pip install ./DPRetrieval
```

</details>

## Running

Use the dataset-specific scripts in the project root. Edit `IMAGE_DIR` and `CUDA_VISIBLE_DEVICES` in the script before running.

```bash
bash run_kitti.sh
bash run_vkitti.sh
bash run_waymo.sh
bash run_openloris.sh
```


You can also call the Python entrypoint directly:

```bash
python vggt_align.py --image_dir ./path/to/images --config ./configs/base_config.yaml
```

To process a video, first extract and resize its frames:

```bash
mkdir -p ./extract_images
ffmpeg -i your_video.mp4 -vf "fps=5,scale=518:-1" ./extract_images/frame_%06d.png
```

Then run:

```bash
python vggt_align.py --image_dir ./extract_images --config ./configs/base_config.yaml --exp_dir ./exps
```

## Datasets

- [Waymo Open Dataset](https://waymo.com/open/) ([v1.4.1](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_1))
- [Virtual KITTI Dataset v1.3.1](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/)
- [KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- OpenLORIS-Scene Dataset

## Acknowledgments

This project builds on [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), [VGGT](https://github.com/facebookresearch/vggt), [DPVO](https://github.com/princeton-vl/DPVO), and [GigaSLAM](https://github.com/DengKaiCQ/GigaSLAM). We thank the authors for making their work publicly available.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026vggt,
  title={VGGT-Align: Bridging Local Reconstruction and Global Consistency for Long-Sequence 3D Reconstruction},
  author={Zhang, Wei and Wu, Yihang and Li, Songhua and Wang, Qi},
  journal={arXiv preprint arXiv:2608.15260},
  year={2026}
}
```

## License

VGGT-Align follows the [VGGT license](./LICENSE.txt). For commercial use, please review the [VGGT repository](https://github.com/facebookresearch/vggt) and the [VGGT-1B-Commercial weights](https://huggingface.co/facebook/VGGT-1B-Commercial).
