该目录用于保存模型学习过程中的中间态数据。

由于文件命名的原因，学习过程中保存的数据仅针对某一张训练图片而言，当有多张训练图片时，后一张训练图片的数据会覆盖掉前一张训练时保存下来的数据。

生成的数据来自`src/learning.py`中的`show_bu_msg()`及`show_frcs()`函数。