# 轴承剩余寿命预测项目(projectRUL)
使用PHM2012大赛的轴承数据库，研究如何使用深度学习算法对滚珠轴承进行剩余寿命预测的试错项目  
to prediction the remain useful life of bearing based on 2012 PHM data
## 2021-01-17 最后更新总结(Final Update)
*碎碎念：这个项目一开始只是一个默默无名的研究生用来记录自己在深度学习算法上的学习过程，所以本质上就是一个没啥卵用的项目。结果项目停止更新的两年多总有人会给我发邮件问这个项目，(也有可能相关方向的代码是真的少，所以找到这里。。。)，而且我也不做相关方向了，所以做一个总结。*
1. 项目里面的深度模型基本都能运行，但是效果都不行！！（仅供新人学习参考）
2. 项目里面我自我感觉最好的代码是`dataset.py`，这是将PHM2012、德国帕德博恩大学的数据库以及cwru数据进行封装，从而让我方便进行数据库替换来验证模型效果。首先要先从网上下载数据库，然后使用`dataset.py`里面的`make_xxx_dataset()`方法生成pkl文件（有些数据量特别大的还是分成了多个pkl文件），加载的时候使用`DataSet.load_dataset()`方法就行了，最后还可以更具条件加载不同的数据，具体看`dataset.py`的代码注释。（**记得看文件(夹)的调用路径是否需要更改**）
3. 最后在预测轴承剩余寿命上，还是有一个效果比较可以的模型可供参考，那就是`attention.py`和`attention2.py`，具体参考论文[A novel deep learning method based on attention mechanism for bearing remaining useful life prediction](https://www.sciencedirect.com/science/article/abs/pii/S1568494619307008)。
4. 之后我的GitHub里面会添加一个ProjectRUL2项目，是我最新也是最后的有关轴承剩余寿命的研究记录。

### English Verision
1. The code in this Repository is able to run, but does not work well!!（For learning Deep Learning Only!s）
2. In this Repository, the most useful code in the one in file`dataset.py`. This code is used to package the dataset from PHM2012, the paderborn, the CWRU and the IMS. First, download the dataset in the network, and use the `make_xxx_dataset()` function in `dataset.py` to generate some pkl files. Then such files are able to load with funciton `DataSet.load_dataset()`, specially select the data according to different condition. More detail please read the code comment.
3. Finally, there is also a useful model based on attention to Predict the bearing RUL. The code in the files `attention.py` and `attention2.py`. More detail can be found in this article: [A novel deep learning method based on attention mechanism for bearing remaining useful life prediction](https://www.sciencedirect.com/science/article/abs/pii/S1568494619307008)。
4. Later, I will upload another Repository called 'ProjectRUL2' in my github. This is my newest and final work log of my researching in Bearing RUL.