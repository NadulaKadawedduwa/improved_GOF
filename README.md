# Installation
Clone the repository and create an anaconda environment using
```
git clone https://github.com/NadulaKadawedduwa/improved_GOF
cd gaussian-opacity-fields

conda create -y -n gof python=3.8
conda activate gof

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# tetra-nerf for triangulation
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
```

# Dataset

Mip-NeRF 360 dataset: [official webiste](https://jonbarron.info/mipnerf360/) (Only bicycle was tested)

Tanks and Temples dataset: [preprocessed here](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main)

For the Tanks and Temples dataset, you need to download the ground truth point clouds, alignments and cropfiles and save to `eval_tnt/TrainingSet`, such as `eval_tnt/TrainingSet/Caterpillar/Caterpillar.ply` from [here](https://www.tanksandtemples.org/download/).


# Training and Evaluation
```
# Parameter studies code on bicycle dataset
python scripts/run_parameter_study.py

# Mip-NeRF 360 Bicycle dataset
python scripts/run_mipnerf360.py

# Tanks and Temples dataset
python scripts/run_tnt.py
```

# Citation
Refer to final report to see what changes were made to the original code.
Codebase was expanded from the original [Gaussian Opacity Fields paper](https://niujinshuchong.github.io/gaussian-opacity-fields/).
```bibtex
@article{Yu2024GOF,
  author    = {Yu, Zehao and Sattler, Torsten and Geiger, Andreas},
  title     = {Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes},
  journal   = {ACM Transactions on Graphics},
  year      = {2024},
}
```
