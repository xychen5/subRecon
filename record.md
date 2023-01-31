# 1 编译 surfel meshing

cuda 10.1(要求切换到 11.7)
gcc 7.5？

```sh
sudo update-alternatives  --install /usr/bin/g++ g++ /usr/bin/g++-7 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50
sudo apt-get install libegl1-mesa-dev
sudo apt-get install libeigen3-dev libceres-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libglew-dev
sudo apt install libqt5x11extras5-dev
mkdir build_RelWithDebInfo
cd build_RelWithDebInfo
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="-arch=sm_61" ..
make -j SurfelMeshing
```

运行：

./build_RelWithDebInfo/applications/surfel_meshing/SurfelMeshing ../dataSets/rgbd_dataset_freiburg1_xyz freiburg1_xyz-rgbdslam.txt

# 2 droid slam

pytorch 1.8.1 -> 1.13.1

```sh
# 1 lietorch
git clone --recursive https://github.com/princeton-vl/lietorch.git
sudo apt-get install python3-dev
sudo apt-get install python3.7-dev
# 切换到10.2，太老，因为torchvision导致1.13.1的torch，导致lietorch要11.7的cuda
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run


pip install --upgrade torch==1.8.1(> 1.7.0 不然这个lietorch编译不过)
sudo vim /usr/include/c++/7/bits/basic_string.tcc +1067
https://github.com/traveller59/spconv/issues/42
# changing devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067 from this:
#
# __p->_M_set_sharable();
#
# to this seems to fix the compiler error:
#
# (*__p)._M_set_sharable();
python setup.py install

```

## 参考本来的 droid slam 的安装

https://github.com/princeton-vl/DROID-SLAM

看到 environment.yaml 里面很多的 dependancy，看着安装一些吧：
pip install torch_scatter
pip install open3d
pip uninstall PIL
pip uninstall Pillow
pip install Pillow
pip install torchvision

考虑https://theairlab.org/tartanair-dataset/的数据集

python demo.py --imagedir=../dataSets/rgbd_dataset_freiburg1_xyz/rgb --calib=../dataSets/rgbd_dataset_freiburg1_xyz/calibration.txt --stride=2 --buffer 384 --mvsnet_ckpt ./data/cds_mvsnet.pth

python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt


# 3 大致行动路线
## 3.0 验证surfelmeshing的构网逻辑

## 3.1 将droid slam得到的相机位姿以及cdm-mvs得到的深度图拼接到 surfelmeshing当中，也就是将surfelmeshing的输入换掉
### 3.1.1 具体拼接策略遇到的难题
- rgbd video是否可以通过这种深度预测加上相机路径来合成？
- 拼接过程的流水线构建？
- 

## 3.2 具体的子场景划分算法完善
- 完善算法
- 涉及到网格融合
- 

# 4 可能能做的对比 -- simple recon : 优缺点: 能够适应室外场景


