# PowerPaint v2-1：人类手臂贴图替换为 G1 外观

这个目录里的方案只做外观重绘：

- 输入是 `dataset/basic_pick_place/{9,10}.mp4`。
- 遮罩复用已经生成并逐帧对齐的 SAM 人类手臂遮罩。
- 推理直接导入 `PowerPaint/` 官方类，官方 checkout 不做任何修改。
- 权重只从 `models/PowerPaint-v2-1` 本地加载，不访问 Hugging Face。
- 使用 PowerPaint 的 `shape-guided` 任务和较高的 `fitting_degree=0.95`，
  保持原来人手/手臂的姿态和轮廓，只把遮罩内改成 G1 的浅灰硬壳、深灰关节和
  黑色机械手外观。
- 最后用遮罩重新合成；有效遮罩外的像素直接来自原始解码帧。

这不是运动学或几何替换方案。人的手指轮廓、手臂长度和物体接触关系会保留，
符合“只改贴图”的试验目的。

## 环境

本机已有 `isaac_sim` 环境的 CUDA PyTorch。下面创建薄的隔离环境并固定
PowerPaint 所需版本，不会改主 `.venv`：

```bash
UV_CACHE_DIR=/tmp/vision-align-uv-cache \
  uv venv --system-site-packages \
  --python /home/ai01/anaconda3/envs/isaac_sim/bin/python \
  draw/.venv_powerpaint

UV_CACHE_DIR=/tmp/vision-align-uv-cache \
  uv pip install \
  --python draw/.venv_powerpaint/bin/python \
  -r draw/requirements-powerpaint.txt
```

## 先验证输入配对

```bash
draw/.venv_powerpaint/bin/python draw/powerpaint_g1_arms.py \
  --validate-only
```

预期：

- episode 9：63 帧；
- episode 10：143 帧；
- 视频和遮罩均为 1920×1080。

## 少量帧预览

```bash
draw/.venv_powerpaint/bin/python draw/powerpaint_g1_arms.py \
  --episodes 9 \
  --frames 0,15,30,45,62
```

结果在：

- `draw/output/9/frames`：重绘帧；
- `draw/output/9/effective_masks`：实际使用的遮罩；
- `draw/output/9/comparisons`：原图 / 红色遮罩 / 结果三联图。

## 完整处理两个视频

```bash
draw/.venv_powerpaint/bin/python draw/powerpaint_g1_arms.py \
  --episodes 9 10
```

完整视频分别输出到：

- `draw/output/9/9_g1_powerpaint.mp4`
- `draw/output/10/10_g1_powerpaint.mp4`

默认同一视频的每帧使用同一个 seed，减少材质颜色跳动。PowerPaint 本身是逐帧
图像模型，所以仍可能出现细节闪烁；本方案刻意不加入光流、机器人运动学或
时序模型。

## 默认提示词

```text
first-person real Unitree G1 robot arms, simple industrial design, smooth matte
ivory-white rounded plastic covers, sparse black seams, dark gray cylindrical
joints, small matte black robotic hands, unlit, photorealistic, same pose, same
silhouette, same perspective, original lighting
```

PowerPaint 的每个任务词会展开成 10 个学习 token；脚本会在推理前检查自定义正、
负提示词都不超过 CLIP 的 77-token 上限。

如结果过于接近人手，可把 `--guidance-scale` 从 `7.5` 提到 `9`。如轮廓被改得
过多，保留默认 `--fitting-degree 0.95`，不要降低它。

## 更稳定的 mesh 贴图方案

PowerPaint v2-1 不支持把 STL/URDF 作为图像条件，所以即便固定 seed，它仍会
在每帧重新设计零件。`mesh_texture_g1_arms.py` 改用实际 G1 mesh：

- 从 `assets/unitree_description/urdf/g1/main_with_hand_new.urdf` 读取左右
  `elbow -> wrist -> hand` 的 STL、零位相对布局以及 white/BlackAccent 材质；
- 只在启动时把零位 mesh 渲成一张固定 atlas；
- 每帧只在二维 SAM mask 中估计根部、主轴和横向位置，把同一 atlas 映射进去；
- atlas 的每一行来自实际 mesh 的横截面，横向拉伸到人臂 mask；不在帧间重新
  生成零件；
- 不计算每帧 G1 关节角，不做 IK、FK、mesh pose fitting。

先看关键帧：

```bash
.venv/bin/python draw/mesh_texture_g1_arms.py \
  --episodes 9 \
  --frames 0,15,30,45,62
```

完整生成：

```bash
.venv/bin/python draw/mesh_texture_g1_arms.py \
  --episodes 9 10
```

结果输出到 `draw/mesh_output`。这个版本优先保证跨帧材质与零件布局稳定；如果
还需要 PowerPaint，可只把它用于窄边界接缝，而不再让它重绘整个手臂内部。
实验性 `--temporal-blend` 默认关闭，因为像素级光流会把黑手/白腕的高对比边界
拖出重影；如需时序平滑，可从 `0.1` 这样的小值开始。

## 去臂背景上插入独立 G1 轮廓

`mesh_insert_g1_arms.py` 才是“先去掉人手，再画 G1”的版本：

- 背景来自 `videos_handless/.../file-{000,001}.mp4`；
- 人类手臂 mask 只估计二维根部、朝向、长度和宽度；
- 完整 G1 mesh sprite 带自己的 alpha 轮廓，绝不被人类 mask 裁剪；
- 只对二维放置参数做平滑，不做机器人关节角、IK 或 FK。

关键帧预览：

```bash
.venv/bin/python draw/mesh_insert_g1_arms.py \
  --episodes 9 \
  --frames 0,15,30,45,62
```

完整视频：

```bash
.venv/bin/python draw/mesh_insert_g1_arms.py --episodes 9 10
```

结果输出到 `draw/insert_output`。该方案保留 G1 自己的掌宽、指形和腕部外轮廓；
因为明确不做 kinematics，手指始终使用 URDF 零位姿态，抓取接触不会物理准确。

## G1 mesh 前臂 + PowerPaint 手部

`hybrid_g1_forearms_powerpaint_hands.py` 是两种方案的混合版：

- 使用无手臂视频作为干净背景；
- 前臂只渲染 G1 URDF 中 elbow/wrist 的四个 link，不载入 hand/finger mesh；
- 只在原手臂末端取上一版 PowerPaint 的手掌和手指，保留抓取手势；
- 腕部设置交叠渐变，人类 mask 不裁剪 G1 前臂外轮廓。

关键帧预览：

```bash
.venv/bin/python draw/hybrid_g1_forearms_powerpaint_hands.py \
  --episodes 9 --frames 0,15,30,45,62
```

生成完整视频：

```bash
.venv/bin/python draw/hybrid_g1_forearms_powerpaint_hands.py \
  --episodes 9 10
```

结果输出到 `draw/hybrid_output`。默认 G1 前臂只延伸到原手臂长度的 `0.82`，
并在 `0.70`–`0.84` 范围淡出；PowerPaint 手部从 `0.68` 开始进入、
`0.80` 完全显示。
