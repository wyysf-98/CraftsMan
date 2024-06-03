## 匠心 —— 最佳实践 :notebook:

### 概览
<p align="center">
  <img src="asset/tutorial_overview.jpg" >
</p>
<font color="red">**CraftsMan (又名 匠心)**</font> 是一个两阶段的文本/图像到3D网格生成模型。通过模仿艺术家/工匠的建模工作流程，我们提出首先使用3D扩散模型生成一个具有平滑几何形状的粗糙网格（5秒），然后使用2D法线扩散生成的增强型多视图法线图进行细化（20秒），这也可以通过类似Zbrush的交互方式进行。

### 基础使用

#### 步骤 1: 上传参考图片
您可以通过点击蓝色区域上传自己的图像，或者您也可以选择页面下方的示例图像进行快速测试。


#### 步骤 2: 点击生成
只需点击`Generate`按钮，我们会首先根据输入生成多视图图像，显示在绿色区域。随后，它将生成网格，这在红色区域展示。
<p align="center">
  <img src="asset/tutorial_generation.jpg" >
</p>

#### 不同的多视角生成模型
图像到MV（多视图）模型在我们的流程中扮演着至关重要的角色，因为它们使我们能够有效地利用2D先验来解决3D生成过程中，只有有限的3D数据的问题。我们提供多种选项，每种模型都有其自己独特的优势和特点。

##### [CRM](https://github.com/thu-ml/CRM)
我们选择CRM作为默认的多视图（MV）模型，由于其轻量级的特性以及与其他选项相比，它与我们的3D扩散模型的分布更为一致。这种一致性通常在大多数情况下都能产生更好的结果。


##### [Era3D](https://github.com/pengHTYX/Era3D)
Era3D 是[Wonder3D](https://github.com/xxlong0/Wonder3D/tree/main)的高级版本，采用更高的分辨率，并引入了一种新颖的方法来解决透视失真问题。这种能力允许它有效管理图像视图处于极端角度的场景。但它需要更长的时间来生成。
<p align="center">
  <img src="asset/tutorial_era3d.jpg" >
</p>

##### [ImageDream](https://github.com/3DTopia/LGM)
ImageDream 最初在 [repo.](https://image-dream.github.io/) 提出, 
但我们使用的是[LGM](https://github.com/3DTopia/LGM)提出的另一个微调版本，它更加稳健，能够产生更精细的结果。

#### 重拓扑
我们使用[Instant Meshes](https://github.com/wjakob/instant-meshes/tree/master)将生成的网格重新网格化为具有2000个面的四边形网格，从而突出显示我们生成的几何形状的平滑性。 通过选择重新网格化选项，您将获得具有规则拓扑结构的重新网格化网格。

<p align="center">
  <img src="asset/tutorial_remesh.jpg" >
</p>

#### 使用文本引导
The [ImageDream](https://github.com/3DTopia/LGM) 还支持使用文本提示作为输入，这显著提高了多视图（MV）预测的详细程度和准确性。这个功能特别适用于某些需要细致背部细节的物体，如皮卡丘的尾巴，可以通过文本引导diffusion生成。

<p align="center">
  <img src="asset/tutorial_text_prompt.jpg" >
</p>

#### 不同的随机数种子

默认情况下，系统使用0作为种子值，表示用于生成的是一个随机种子。为了获得更一致和可预测的结果，您可以选择为生成过程设置特定的种子。

#### 获得更好结果的小技巧
- 尝试使用对象的正面视图图像；
- 使用不同的文本到MV模型；
- 尝试不同的参数；

### 高级选项

#### Only Generate 2D
您可以通过此按钮仅生成多视图图像。

#### Only Generate 3D
您可以通过此按钮仅生成网格，请务必首先生成或上传多视图图像。

### 高级选项 (2D)

#### Backgroud Choice
`Alpha as Mask`: 使用图像的透明通道作为遮罩。
`Auto Remove Background`: 使用不同的方法自动去除背景。
`Original Image`: 使用原始图像。

#### Backgroud Remove Type
我们提供两种方式[rembg](https://github.com/danielgatis/rembg) 和 [sam](https://github.com/facebookresearch/segment-anything)来移除背景颜色。SAM可能需要更多的处理时间，但在前景分割方面提供更高的准确性。如果您在使用默认的rembg时遇到获取正确前景的问题，我们建议尝试SAM。

#### Backgroud Color
对于CRM模型，建议使用灰色背景颜色，我们已经手动将其设置为默认。对于其余的多视图（MV）模型，白色背景在大多数场景中就足够了。

### 高级选项 (3D)

#### Octree Depth
Marching Cubes算法使用一个分辨率参数；7的值对应于2^7个体素。为生成更精确的网格，请相应调整此值。

### Q&A

#### Q1. 为什么输出网格似乎与参考图像不完全一致？
我们的3D扩散是一个生成模型，它以图像作为参考（与从稀疏视图构建3D网格的重建模型相比），因此生成的网格与输入的不完全对齐。
为解决这个问题，您可以尝试以下方法：
使用对象的正面视图图像作为输入，这样您将获得更准确的多视图预测；
尝试另一个种子和mv_models，例如ImageDream倾向于产生轴对齐的多视图图像；
我们将通过引入更准确的条件特征，在未来的工作中增加生成网格与输入图像之间的对齐度。

#### Q2. 为什么输出网格似乎没有纹理？

嗨，我们目前的工作重点是高质量的网格生成，而没有太多关注纹理。我们将在下一个项目中考虑这个方向。目前，可以尝试下面生成纹理的工作来获得纹理: [DreamMat](https://github.com/zzzyuqing/DreamMat?tab=readme-ov-file), [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper), [SyncMVD](https://github.com/LIU-Yuxin/SyncMVD), [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) 等.
