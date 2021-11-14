# RCN_for_CAPTCHA

本项目来自于本人本科毕业设计工作的部分章节内容，可以作为一个独立的整体，因此整理后开源出来。

本项目基于论文*A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs*及其开源代码展开研究，基本实现了原论文中提出的使用递归皮层网络（Recursive Cortical Network，RCN）模型进行文字验证码图片识别的过程。

模型的详细原理请参考原论文。

本项目代码使用`Python 3.6`，程序入口为`src/run_for_MultiObjects.py`

参考资料：
- 参考文献地址：[A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs](https://science.sciencemag.org/content/358/6368/eaag2612.long)
- 参考文献开源代码地址：[science_rcn](https://github.com/vicariousinc/science_rcn)
