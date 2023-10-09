# SMOKE 3D目标检测

| 编辑人           | 编辑时间  | 版本 |
| ---------------- | --------- | ---- |
| @恩培-计算机视觉 | 2023-5-20 | V1.0 |

[toc]

## 一、导出ONNX，build engine

```bash
# （由于目前没有靠谱container，这里用了一个container然后替换安装一些依赖库。如果可以自行配置环境，也可以，请follow mmdetection3d 1.0.0rc0版本的安装指引。）

# pull 镜像
docker run --gpus all -v `pwd`:/app  --rm -it  lilydedbb/mmdetection3d:vision-20211231_1109

# 进入/mmdetection3d目录
cd /mmdetection3d

# 获取代码
git fetch
git checkout v1.0.0rc0

# 调整依赖库版本
pip install mmsegmentation==0.20.0
pip install mmdet==2.19.0
pip install mmcv-full==1.4.0
pip install -v -e .


# 将附件中文件smoke_mono3d_head.py、smoke_pth2onnx.py 拷贝到/mmdetection3d
# 移动
mv smoke_mono3d_head.py mmdet3d/models/dense_heads/smoke_mono3d_head.py

# 创建权重文件夹
mkdir /mmdetection3d/checkpoints
# 将附件中的权重smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth 移到/mmdetection3d/checkpoints中

# 执行ONNX导出命令
python smoke_pth2onnx.py


# 进入项目文件夹
cmake -B build
cmake --build build
./build/build --onnx_file ./weights/smoke_dla34.onnx
./build/smoke_test --smoke ./weights/smoke_dla34.engine --vid media/test.mp4 
```

## 二、SOMKE后处理decode过程

> 参考论文：https://arxiv.org/pdf/2002.10111.pdf
>
> 参考仓库：https://github.com/open-mmlab/mmdetection3d
>
> 自定义算子plugin位置：https://github.com/open-mmlab/mmdeploy/tree/main/csrc/mmdeploy/backend_ops/tensorrt



SOMKE的网络输入大小是`[1,3,384,1280]`，输出节点如下：

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202305222140280.png?x-oss-process=style/wp)

* 1123：`[100,8]`，负责预测检测框相关信息
* 1125：`[100]`，负责输出置信度
* 1126：`[1,100]`，负责输出类别

1123节点输出具体检测框的信息为$\tau = [\delta_z,\delta_{x_c},\delta_{y_c},\delta_h,\delta_w,\delta_l,sin\ \alpha,cos\ \alpha]^T$，具体释义如下：

* $\delta_z$：深度偏移量
* $\delta_{x_c}$：下采样后关键点坐标x方向的偏移量
* $\delta_{y_c}$：下采样后关键点坐标y方向的偏移量
* $\delta_h,\delta_w,\delta_l$：目标体积的残差（高度、宽度、长度）
* $sin\ \alpha,cos\ \alpha$：目标旋转角的向量化表示



解码过程：

1. 深度信息$z$可以根据预先设定的缩放和偏移系数恢复：

$$
z=\mu_z + \delta_z \sigma_z
$$



```C++
// Depth
base_depth_ = {28.01f, 16.32f};
float z = base_depth_[0] + bbox_preds[8 * i] * base_depth_[1];
```

2. 假设$[x\ y\ z]^T$是物体3D检测框的中心，$[x_c\ y_c]^T$是3D中心点在图像平面上的投影点（区别如下图所示），$K^{-1}_{3\times3}$是相机内参矩阵的逆矩阵，则有

   > 可参考《视觉SLAM十四讲》第5.1节相机模型
   >
   > ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202305232257679.png?x-oss-process=style/wp)

   $$
   \begin{bmatrix}
   x \\
   y \\
   z
   \end{bmatrix}
   =K^{-1}_{3\times 3}
   \begin{bmatrix}
   z \cdot x_c \\
   z \cdot y_c \\
   z
   \end{bmatrix}
   $$

   

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202305222158158.png?x-oss-process=style/wp" style="zoom:33%;" />

> 红色的是2D检测框的中心，橙色的是3D检测框中心。

由于网络中进行了特征图下采样，所以需要加上偏移系数
$$
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
=K^{-1}_{3\times 3}
\begin{bmatrix}
z \cdot (x_c + \delta_{x_c}) \\
z \cdot (y_c + \delta_{y_c}) \\
z
\end{bmatrix}
$$

```C++
// https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
int class_id = static_cast<int>(topk_indices[i] / kOutputH / kOutputW); 
int location = static_cast<int>(topk_indices[i]) % (kOutputH * kOutputW);
int img_x = location % kOutputW; 
int img_y = location / kOutputW;

// Location
cv::Mat img_point(3, 1, CV_32FC1); // 
img_point.at<float>(0) = 4.0f * (static_cast<float>(img_x) + bbox_preds[8 * i + 1]);
img_point.at<float>(1) = 4.0f * (static_cast<float>(img_y) + bbox_preds[8 * i + 2]);
img_point.at<float>(2) = 1.0f;
cv::Mat cam_point = intrinsic_.inv() * img_point * z;
float x = cam_point.at<float>(0);
float y = cam_point.at<float>(1);
```



3. 计算目标体积$[h\ w \ t]^T$，首先对整个数据集计算各个类别目标的平均体积，然后基于前面提到的网络预测的体积残差计算目标的真实体积如下：

$$
\begin{bmatrix}
h \\
w \\
l
\end{bmatrix}
=
\begin{bmatrix}
\bar h \cdot e^{\delta_h} \\
\bar w \cdot e^{\delta_w} \\
\bar l \cdot e^{\delta_l} \\

\end{bmatrix}
$$

```C++
// Dimension
base_dims_.resize(3); // pedestrian, cyclist, car
base_dims_[0].x = 0.88f;
base_dims_[0].y = 1.73f;
base_dims_[0].z = 0.67f;
base_dims_[1].x = 1.78f;
base_dims_[1].y = 1.70f;
base_dims_[1].z = 0.58f;
base_dims_[2].x = 3.88f;
base_dims_[2].y = 1.63f;
base_dims_[2].z = 1.53f;

float w = base_dims_[class_id].x * expf(Sigmoid(bbox_preds[8 * i + 3]) - 0.5f);
float l = base_dims_[class_id].y * expf(Sigmoid(bbox_preds[8 * i + 4]) - 0.5f);
float h = base_dims_[class_id].z * expf(Sigmoid(bbox_preds[8 * i + 5]) - 0.5f);
```



4. 计算偏航yaw角度$\theta$
   $$
   \theta = \alpha_z + arctan(\frac{x}{z})
   $$
   

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img202305222313838.png?x-oss-process=style/wp" style="zoom:33%;" />

```c++
// Orientation
float ori_norm = sqrtf(powf(bbox_preds[8 * i + 6], 2.0f) + powf(bbox_preds[8 * i + 7], 2.0f));
bbox_preds[8 * i + 6] /= ori_norm; // sin(alpha)
bbox_preds[8 * i + 7] /= ori_norm; // cos(alpha)
float ray = atan(x / (z + 1e-7f));
float alpha = atan(bbox_preds[8 * i + 6] / (bbox_preds[8 * i + 7] + 1e-7f));
if (bbox_preds[8 * i + 7] > 0.0f)
{
  alpha -= M_PI / 2.0f;
}
else
{
  alpha += M_PI / 2.0f;
}

float angle = alpha + ray;

if (angle > M_PI)
{
  angle -= 2.0f * M_PI;
}
else if (angle < -M_PI)
{
  angle += 2.0f * M_PI;
}
```

5. 根据旋转矩阵$R_\theta$，中心点位置$[x\ y\ z]^T$，目标体积$[h\ w \ t]^T$，可以得到最终的8个点的坐标：
   $$
   B = R_\theta
   \begin{bmatrix}
   \pm h/2 \\
   \pm w/2 \\
   \pm l/2 \\
   \end{bmatrix}
   +
   \begin{bmatrix}
   x \\
   y \\
   z \\
   \end{bmatrix}
   $$

```C++
// https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
//              front z
//                   /
//                  /
//    (x0, y0, z1) + -----------  + (x1, y0, z1)
//                /|            / |
//               / |           /  |
// (x0, y0, z0) + ----------- +   + (x1, y1, z1)
//              |  /      .   |  /
//              | / origin    | /
// (x0, y1, z0) + ----------- + -------> x right
//              |             (x1, y1, z0)
//              |
//              v
//         down y
cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << -w, -l, -h, // (x0, y0, z0)
                       -w, -l, h,                           // (x0, y0, z1)
                       -w, l, h,                            // (x0, y1, z1)
                       -w, l, -h,                           // (x0, y1, z0)
                       w, -l, -h,                           // (x1, y0, z0)
                       w, -l, h,                            // (x1, y0, z1)
                       w, l, h,                             // (x1, y1, z1)
                       w, l, -h);                           // (x1, y1, z0)
cam_corners = 0.5f * cam_corners;
cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
rotation_y.at<float>(0, 0) = cosf(angle);
rotation_y.at<float>(0, 2) = sinf(angle);
rotation_y.at<float>(2, 0) = -sinf(angle);
rotation_y.at<float>(2, 2) = cosf(angle);
// cos, 0, sin
//   0, 1,   0
//-sin, 0, cos
cam_corners = cam_corners * rotation_y.t(); // 得到旋转后的坐标
for (int i = 0; i < 8; ++i)
{
  cam_corners.at<float>(i, 0) += x;
  cam_corners.at<float>(i, 1) += y;
  cam_corners.at<float>(i, 2) += z;
}
cam_corners = cam_corners * intrinsic_.t(); // 得到在图像上的坐标
std::vector<cv::Point2f> img_corners(8);
for (int i = 0; i < 8; ++i)
{
  img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
  img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
}
corners_vec.push_back(img_corners);
```

