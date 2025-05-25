# PTZ-Calib

**PTZ-Calib** is a robust two-stage PTZ camera calibration method, that efficiently and accurately estimates camera parameters for arbitrary viewpoints.
Our method includes an offline and an online stage. In the offline stage, we first uniformly select a set of reference images that sufficiently overlap to encompass a complete 360◦
view. We then utilize the novel PTZ-IBA (PTZ Incremental Bundle Adjustment) algorithm to automatically calibrate the cameras within a local coordinate system. Additionally, for
practical application, we can further optimize camera parameters and align them with the geographic coordinate system using extra global reference 3D information. 
In the online stage, we formulate the calibration of any new viewpoints as a relocalization problem.

![Pipeline of the two-stage PTZ-Calib method](/figures/pipeline.jpg)

## Build

The program is compiled in an Ubuntu 20.04 environment with g++ 9.4.0. You can use the following scripts to install dependencies and compile the program separately.

```bash
# Install dependencies
bash install_deps.sh install_all_deps

# Build program
bash build.sh
```

## Dataset download

You can download both Synthetic and WorldCup14 Dataset from [this link](https://drive.google.com/file/d/1vc7IZJl8-vduPSp41RLDx8nm8ncxzn3b/view?usp=sharing). Note that we only use a subset of the original [WorldCup14 Dataset](https://nhoma.github.io/data/soccer_data.tar.gz) in our experiment. The urban scene dataset used in the paper is temporarily unavailable due to data security and other concerns.

## Run PTZ-BA (offline-stage)

We provide `run_ptzba_synthetic.sh` and `run_ptzba_worldcup14.sh` for batch running PTZ-BA on two datasets. You can check the scripts for more details.

## Run PTZ-Reloc (online-stage)

We provide `run_reloc_synthetic.sh` and `run_reloc_worldcup14.sh` for batch running PTZ-Reloc on two datasets. You can check the scripts for more details.

## Citation

If you use any of this code, please cite our [paper](https://arxiv.org/pdf/2502.09075). This paper was accepted by ICRA 2025.

```bibtex
@article{guo2025ptzcalib,
  title={PTZ-Calib: Robust Pan-Tilt-Zoom Camera Calibration},
  author={Jinhui Guo and Lubin Fan and Bojian Wu and Jiaqi Gu and Shen Cao and Jieping Ye},
  journal={arXiv preprint arXiv:2502.09075},
  year={2025}
}
```
