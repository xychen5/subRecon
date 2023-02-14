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

./build_RelWithDebInfo/applications/surfel_meshing/SurfelMeshing ../dataSets/rgbd_dataset_freiburg1_xyz freiburg1_xyz-rgbdslam.txt \
--export_mesh "res.obj" \
--export_point_cloud "res2.ply"

# 2 droid slam(直接使用 conda 安装是最好的)

本节的自己装环境不推荐
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

## 3.0 验证 surfelmeshing 的构网逻辑
- 构造网格是实时的（每一帧对surfel的变动都可以反应到网格上），但是思考这么一个问题，对于同一个物体，第i帧和i+n帧都拍到，那么surfel会不会需要更新？
  - 那么也没关系，毕竟前面的slam部分我们只提供深度图，而不是基于surfel的slam涉及到surfel的更新！！！也就是我们采用了droidslam和cds mvsnet的深度预测替代了这个基于surfel的slam
> 那么毕设实主要的贡献在于，当场景过大，就采用子场景的方式在重建和渲染方面提速？关键是重建部分减少涉及到的场景的调入调出是否能够加速，也就是说得进一步了解surfel的稠密重建和场景规模的关系，而后修改一下这一部分
### 3.0.1 surfelmeshing的稠密和场景规模的关系探索清楚？
大致发现是全放在gpu里的，进一步确定需要看集成每一帧怎么集成的

## 3.1 将 droid slam 得到的相机位姿以及 cdm-mvs 得到的深度图拼接到 surfelmeshing 当中，也就是将 surfelmeshing 的输入换掉
### 3.1.1 具体拼接策略遇到的难题

- rgbd video 是否可以通过这种深度预测加上相机路径来合成？
- 拼接过程的流水线构建？
- 

## 3.2 具体的子场景划分算法完善

- 完善算法
- 涉及到网格融合
-

# 4 可能能做的对比 -- simple recon : 优缺点: 能够适应室外场景

# 5 毕设的些许思考
## 5.1 展望部分
- 考虑surfel的平滑去噪


# 6 开题报告需要更改的地方：
- 2.3.4（3）深度测量集成，公式2错了，公式3有些不对

# 附录
# 1 cuda基础
- dim3是NVIDIA的CUDA编程中一种自定义的整型向量类型，基于用于指定维度的uint3。例如：dim3 grid（num1，num2，num3）；dim3类型最终设置的是一个三维向量，三维参数分别为x,y,z;
- paged locked mem: 锁页就是将内存页面标记为不可被操作系统换出的内存。所以设备驱动程序给这些外设编程时，可以使用页面的物理地址直接访问内存（DMA），从而避免从外存到内存的复制操作。CPU 仍然可以访问上述锁页内存，但是此内存是不能移动或换页到磁盘上的。CUDA 中把锁页内存称为pinned host memory 或者page-locked host memory。
- 
