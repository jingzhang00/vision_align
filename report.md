# EgoDex / Nymeria VDA Pipeline 总结

## 总览

- **EgoDex**：`HDF5 + MP4` → 人体/手部重定向为 51-DoF G1 → 去除原视频中的人手/手臂 → VDA 深度 → RGB-D scene mesh + G1 在 Isaac 中逐帧回放。
- **Nymeria**：`MP4 + 208-D motion + SMPL metadata + Aria MPS` → 对齐相机和人体轨迹 → 重定向为 51-DoF G1 → 去手臂、VDA 深度、Isaac 回放。
- batch 模式按 episode 独立处理，跳过已完成结果，并在结束后删除中间目录。
- 51-DoF 由 29 个 G1 body DoF 和 22 个 hand DoF 组成。

## 坐标约定

全文固定使用下列记号，不在不同公式中复用含义：

| 符号 | 唯一含义 |
|---|---|
| `E` | EgoDex HDF5 source world |
| `A` | Aria MPS world |
| `M` | Nymeria 208-D motion world（原生 Y-up） |
| `W` | G1 / Isaac robot world：X 前、Y 左、Z 上，单位米 |
| `S` | shared 阶段的 source world；EgoDex 时等于 `E`，Nymeria 时等于 `A` |
| `D` | Aria device frame |
| `H` | Nymeria 208-D motion head frame |
| `P` | source human spine3 frame |
| `J` | 任意人体部件 frame；`J_hip` 固定表示人体 hip frame |
| `G` | G1 torso frame |
| `C_src` | 原始/源 OpenCV optical camera：X 右、Y 下、Z 前 |
| `L_cap` | G1 D435 capture camera-link：X 前、Y 左、Z 上 |
| `C_cap` | G1 D435 capture OpenCV optical camera |
| `k` | 帧索引；`0` 表示首帧 |

位姿、旋转、平移和点统一记为：

$$
\mathbf p_U=\mathbf R^U_V\mathbf p_V+\mathbf t^U_V,
\qquad
\mathbf T^U_V=
\begin{bmatrix}
\mathbf R^U_V&\mathbf t^U_V\\
\mathbf 0^T&1
\end{bmatrix}
$$

参数：`U` 是目标坐标系，`V` 是源坐标系；`T`、`R`、`t` 分别表示完整刚体变换、旋转矩阵和平移，`p` 表示三维点。因此 camera-to-world 是 `T^S_Csrc`，world-to-camera 始终是其逆。

文件中的位姿四元数统一为 `XYZW`；调用 Isaac API 时重排为 `WXYZ`。capture camera-link 到 OpenCV optical 的固定轴变换为：

$$
\mathbf R^{L_{cap}}_{C_{cap}}=
\begin{bmatrix}
0&0&1\\
-1&0&0\\
0&-1&0
\end{bmatrix}
$$

参数：矩阵三列分别是 `C_cap` 的 X、Y、Z 轴在 `L_cap` 中的方向。

## EgoDex：原始 camera 与人体轨迹

1. 从 HDF5 读取相机内参、逐帧 camera-to-world、spine3-to-world 以及 hip、双臂、双手变换；MP4 与 HDF5 必须逐帧一致。人体关键点位置使用 7 帧二阶 Savitzky–Golay 平滑。
2. EgoDex hip 局部轴依次为 X 左、Y 上、Z 前。首帧把它重排成 G1 的 X 前、Y 左、Z 上，并用 SVD 投影到最近的合法旋转：

$$
\boldsymbol\Pi=
\begin{bmatrix}
0&1&0\\
0&0&1\\
1&0&0
\end{bmatrix},
\qquad
\mathbf R^W_E=
\left[
\operatorname{Proj}_{SO(3)}
\left(\mathbf R^E_{J_{hip}}(0)\boldsymbol\Pi\right)
\right]^T
$$

参数：`Pi` 始终是 `[前,左,上]` 的固定列重排；`R^E_Jhip(0)` 是首帧 EgoDex hip 旋转；`Proj_SO(3)` 表示取最近的右手正交旋转；`R^W_E` 始终是 EgoDex-world → G1-world 固定旋转。后续人体转身不会改变这个世界基。

3. 动态目标主要来自双臂/双手：下肢、躯干和肩部位置采用 G1 neutral standing；源肩—肘、肘—腕方向经 `R^W_E` 映射后驱动 GMR。手腕使用 tracked Hand 位姿和左右手固定轴映射；有 25 点手部轨迹时独立求 22 个 O6 手指关节，否则保持张开。
4. GMR 输出 29 个 body DoF，和 22 个手指 DoF 合并为 51-DoF G1；root 位姿由 GMR free joint 提取，四元数由 `WXYZ` 改存为 `XYZW`。

## Nymeria：原始 camera 与人体轨迹

1. 用 MP4 的实际显示时间戳对 208-D motion 重采样：位置线性插值、旋转用 Slerp；root 位移逐帧积分。208-D 原生为 Y-up，送入 GMR 前转为 G1 的 Z-up：

$$
\mathbf p_W(k)=
\begin{bmatrix}
z_M(k)&x_M(k)&y_M(k)
\end{bmatrix}^T,
\qquad
\mathbf q^W_M=[0.5,0.5,0.5,0.5]_{XYZW}
$$

参数：`x_M,y_M,z_M` 是点在 `M` 中的原始分量；`p_W` 是同一点转换到 G1 Z-up 世界轴后的坐标；`q^W_M` 始终是固定的 M → W 轴旋转四元数。

2. 原始相机轨迹来自 Aria closed-loop SLAM，camera-to-device 外参来自在线标定；两者加入视频顺时针旋转 90°对应的 optical-frame 修正后组合为：

$$
\mathbf T^A_{C_{src}}(k)=
\mathbf T^A_D(k)\mathbf T^D_{C_{src}}
$$
作用：将nymeria clip的OpenCV光轴optical frame对齐到Aria camera-rgb标定frame
参数：`T^A_D(k)` 是 SLAM device 轨迹；`T^D_Csrc` 是包含 upright-video 90°轴修正的固定 camera-to-device 外参；结果 `T^A_Csrc(k)` 是源相机在 Aria 世界中的位姿。

3. Aria device 与 208-D head 的轨迹位于两个独立世界，但头显固定在头部。先在相同的两个时刻 `i,j` 构造各自的相对运动：

$$
\begin{aligned}
\Delta\mathbf T_D(i,j)
&=\left(\mathbf T^A_D(i)\right)^{-1}\mathbf T^A_D(j),\\
\Delta\mathbf T_H(i,j)
&=\left(\mathbf T^M_H(i)\right)^{-1}\mathbf T^M_H(j)
\end{aligned}
$$

参数：`Delta T_D` 始终表示 device 相对运动，`Delta T_H` 始终表示 208-D head 相对运动；`i,j` 是同一 clip 内的两个采样帧。

session 级 hand-eye 标定求唯一的固定 head-to-device 外参，使两条相对运动链一致：

$$\Delta\mathbf T_D(i,j)\mathbf T^D_H=\mathbf T^D_H\Delta\mathbf T_H(i,j)$$

参数：`T^D_H` 始终表示 motion-head → Aria-device 的固定安装变换；同一 session 的所有可用 clip 共同估计它，但不跨 clip 构造相对运动。

   求得外参后，每帧把全部人体部件注册到 Aria 世界：

$$
\begin{aligned}
\mathbf T^A_M(k)
&=\mathbf T^A_D(k)\mathbf T^D_H
\left(\mathbf T^M_H(k)\right)^{-1},\\
\mathbf T^A_J(k)
&=\mathbf T^A_M(k)\mathbf T^M_J(k)
\end{aligned}
$$

参数：`T^A_M(k)` 是每帧 motion-world → Aria-world 对齐；`T^M_J(k)` 和 `T^A_J(k)` 分别是任意人体部件对齐前、后的位姿。

   因而输出 HDF5 中的 camera、hip、spine3、双臂和双手共享同一个 Aria MPS 世界坐标。
4. 完整 22-joint SMPL-X 轨迹经 Y-up → Z-up 后送入 GMR，并按参与者身高缩放。GMR 产生 29 个 body DoF；Nymeria 没有手指观测，所以另外 22 个手指关节固定在 URDF neutral/open 值。
5. 首帧建立 Aria-world → G1-world 的固定旋转：

$$
\mathbf R^W_A=
\mathbf R^W_{J_{hip},208D}(0)
\left(\mathbf R^A_{J_{hip},aligned}(0)\right)^T
$$

参数：第一项是首帧 208-D hip 转成 G1 Z-up 后的旋转，第二项是同一 hip 在 Aria 世界中的对齐旋转；转置是旋转矩阵的逆，结果 `R^W_A` 始终表示 Aria-world → G1-world 的固定旋转。

## Retarget 后的 scene 与机器人轨迹

两条 pipeline 从这里开始共用同一条处理链。

### 1. 虚拟 scene-camera

先用 G1 root + 51 个关节做 FK，得到每帧 torso 位姿；再计算源相机在人体 spine3 下的相对位姿：

$$
\mathbf T^P_{C_{src}}(k)=
\left(\mathbf T^S_P(k)\right)^{-1}
\mathbf T^S_{C_{src}}(k)
$$

参数：`T^S_P(k)` 是 source spine3 位姿，`T^S_Csrc(k)` 是原始 source camera 位姿；结果 `T^P_Csrc(k)` 始终表示源相机在 spine3 局部坐标中的位姿。这里沿用全局定义：`P` 只表示 spine3，`H` 只表示 Nymeria motion head。

scene-camera 的**位置**锚定到 G1 torso，**旋转**保留源相机的世界旋转：

$$
\begin{aligned}
\mathbf t^W_{C_{scene}}(k)
&=\mathbf t^W_G(k)
+s_{scene}\mathbf R^W_G(k)\mathbf B
\mathbf t^P_{C_{src}}(k),\\
\mathbf R^W_{C_{scene}}(k)
&=\mathbf R^W_S\mathbf R^S_{C_{src}}(k)
\end{aligned}
$$

参数：`C_scene` 只表示放置 RGB-D mesh 的虚拟相机；`t^W_G,R^W_G` 是 G1 torso 的 FK 位姿；`s_scene` 始终是 GMR spine3 scene scale；`B` 是 source spine3 局部轴 → G1 torso 局部轴的固定旋转；`R^W_S` 是 source-world → G1-world 固定旋转。EgoDex 时 `S=E`，Nymeria 时 `S=A`。

固定局部轴旋转只由首帧计算：

$$
\mathbf B=
\left(\mathbf R^W_G(0)\right)^T
\mathbf R^W_S\mathbf R^S_P(0)
$$

参数：`R^W_G(0)` 是首帧 G1 torso 旋转，`R^S_P(0)` 是首帧 source spine3 旋转；`B` 在整段 episode 中保持不变。

这样 scene 的尺度、位置跟随机器人，但仍保留原始相机的朝向变化。

EgoDex 还用前 5 帧双手掌计算一个常量 scene 平移，取“机器人手掌与缩放后源手掌”世界残差的中位数；Nymeria 不增加这项平移。

### 2. RGB-D scene

- SAM 3 跟踪原视频中的人手/手臂，ProPainter 补全背景。
- Metric Video Depth Anything 从补全视频估计深度，并把内参同步缩放到 mesh 分辨率。深度缩放为：

$$
z_{mesh}(k,u,v)=
z_{VDA}(k,u,v)\,s_{VDA}\,s_{scene}
$$

参数：`z_VDA` 始终是模型原始深度，`s_VDA` 始终是 episode 级 metric 标定比例，`s_scene` 与上一节相同，`z_mesh` 是进入 mesh 的最终米制深度；`u,v` 是像素坐标。

- 每帧按 OpenCV 模型反投影：

$$
\mathbf p_{C_{scene}}(k,u,v)=
z_{mesh}(k,u,v)
\begin{bmatrix}
(u-c_x)/f_x&(v-c_y)/f_y&1
\end{bmatrix}^T
$$

参数：`p_Cscene` 是虚拟 scene-camera 坐标中的三维点；`f_x,f_y,c_x,c_y` 始终来自缩放到 mesh 分辨率后的 source camera 内参。

  再变换到 G1 世界：

$$
\mathbf p_W(k,u,v)=
\mathbf R^W_{C_{scene}}(k)\mathbf p_{C_{scene}}(k,u,v)
+\mathbf t^W_{C_{scene}}(k)+\boldsymbol\Delta_{scene}
$$

参数：`p_W` 是 G1 world 中的 scene 点；`Delta_scene` 始终是常量 scene 平移：EgoDex 由前 5 帧双手掌残差的世界中位数得到，Nymeria 为零。

- 相邻像素在深度连续时组成三角面，补全后的 RGB 作为纹理；因此 scene 是逐帧更新的 camera-centric RGB-D mesh。

### 3. Isaac 中的机器人与最终相机

- 每帧把 G1 root 位姿和 51 个关节角写入 Isaac articulation，并回读校验后渲染。
- `batch_egodex.sh` 和 `batch_nymeria.sh` 都使用 `--capture-view robot-camera`。最终成片相机由 G1 torso FK、固定 D435 安装外参和固定 camera-link → OpenCV optical 轴变换组成：

$$
\mathbf T^W_{C_{cap}}(k)=
\mathbf T^W_G(k)
\mathbf T^G_{L_{cap}}
\mathbf T^{L_{cap}}_{C_{cap}}
$$

参数：`T^G_Lcap` 始终是 D435 相对 G1 torso 的固定安装位姿；`T^Lcap_Ccap` 的平移为零，旋转即“坐标约定”中的固定矩阵；结果 `T^W_Ccap(k)` 是最终 capture camera 位姿。

- **关键区别**：上面的 scene-camera 只负责把 RGB-D mesh 放入世界；最终 D435 capture-camera 是独立的机器人头部相机。渲染内参使用按输出尺寸等比缩放的 D435 标定，而不是源视频内参。
