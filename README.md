![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** 是一个GPU加速的开源框架，旨在统一和简化机器人研究工作流程，如强化学习、模仿学习和运动规划。基于[NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)构建，它结合了快速且准确的物理和传感器模拟，使其成为机器人从模拟到现实转移的理想选择。

Isaac Lab为开发者提供了一系列用于精确传感器模拟的基本功能，如基于RTX的相机、激光雷达或接触传感器。该框架的GPU加速使用户能够更快地运行复杂的模拟和计算，这对于强化学习等迭代过程和数据密集型任务至关重要。此外，Isaac Lab可以在本地运行或分布在云端，为大规模部署提供灵活性。

## 主要特点

Isaac Lab提供了一套全面的工具和环境，旨在促进机器人学习：
- **机器人**：多样化的机器人集合，从机械臂、四足机器人到人形机器人，包含16种常见可用模型。
- **环境**：超过30个可直接训练的环境实现，可以使用流行的强化学习框架如RSL RL、SKRL、RL Games或Stable Baselines进行训练。我们还支持多智能体强化学习。
- **物理**：刚体、关节系统、可变形物体
- **传感器**：RGB/深度/分割相机、相机标注、IMU、接触传感器、射线投射器。


## 入门指南

我们的[文档页面](https://isaac-sim.github.io/IsaacLab)提供了您开始所需的一切，包括详细教程和分步指南。点击以下链接了解更多信息：

- [安装步骤](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [强化学习](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [教程](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [可用环境](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## 为Isaac Lab做贡献

我们热烈欢迎社区的贡献，使这个框架成熟并对每个人都有用。
这些贡献可以是错误报告、功能请求或代码贡献。有关详细信息，请查看我们的
[贡献指南](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)。

## 展示与分享：分享您的灵感

我们鼓励您使用我们在此存储库的`讨论`部分中的[展示与分享](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)区域。这个空间旨在让您：

* 分享您创建的教程
* 展示您的学习内容
* 展示您开发的令人兴奋的项目

通过分享您的工作，您将激励他人并为我们社区的集体知识做出贡献。您的贡献可以激发新的想法和合作，促进机器人和模拟领域的创新。

## 故障排除

请查看[故障排除](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html)部分了解常见修复方法或[提交问题](https://github.com/isaac-sim/IsaacLab/issues)。

对于与Isaac Sim相关的问题，我们建议查看其[文档](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)或在其[论坛](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67)上提问。

## 支持

* 请使用GitHub [讨论](https://github.com/isaac-sim/IsaacLab/discussions)讨论想法、提问和请求新功能。
* GitHub [问题](https://github.com/isaac-sim/IsaacLab/issues)应仅用于跟踪具有明确范围和明确可交付成果的可执行工作项。这些可以是修复错误、文档问题、新功能或一般更新。

## 与NVIDIA Omniverse社区联系

有想要更广泛分享的项目或资源吗？我们很乐意听取您的意见！请联系NVIDIA Omniverse社区团队，邮箱为OmniverseCommunity@nvidia.com，讨论更广泛传播您工作的潜在机会。

加入我们，共同构建一个充满活力、协作的生态系统，让创造力和技术相交。您的贡献可以对Isaac Lab社区及更广泛领域产生重大影响！

## 许可证

Isaac Lab框架在[BSD-3许可证](LICENSE)下发布。`isaaclab_mimic`扩展及其相应的独立脚本在[Apache 2.0](LICENSE-mimic)下发布。其依赖项和资产的许可证文件位于[`docs/licenses`](docs/licenses)目录中。

## 致谢

Isaac Lab的开发源自[Orbit](https://isaac-orbit.github.io/)框架。我们将感谢您在学术出版物中引用它：
