# Multi-process description

专门实现[base_store.py](../nebula_pyg/base_store.py)的原因，是在为`FeatureStroage`的`get_tensor()`做`query`的时候发现的，当时遇到了一个情况，我需要使用query从nebula-graph中获取指定index列表的属性。（TODO：可以写一个关于这个的内容）

先插入一句，简单介绍一下，PyG的`DataLoader`允许进行`num_workers`的设置，这个应该是从[PyTorch](https://docs.pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)中继承而来的。

在`get_tensor()`的具体实现中，最初的实现形式是由用户初始化一个`gclient`(连接graphD的session)，和`sclient`(连接storageD的session)，后续的所有操作都通过这一个`gclient`和`sclient`进行。但当引入query了以后，session的操作就变得极为频繁（其实本来就应该很频繁）。

很可惜，没有找到官方的说明，通过别人的blog，大致可以把torch的multi-process dataloader理解为，由主进程维护一个`index queue`，worker从中取出，调用`get_tensor()`，之后将返回的数据放到 worker_result_queue 这个队列供主进程访问，主进程统一处理。

最开始的实现很简单：用户初始化一个 gclient（连 graphd）和 sclient（连 storaged），后续所有的操作都是通过这两个对象。

这个时候问题就出来了，如果是多进程进行，那么get_tensor()中的 session 在最初的设计中都是从上层统一传递的(唯一对象)，并且[nebulagraph的session并不安全](https://www.nebula-graph.com.cn/posts/informal-analysis-of-session-in-nebulagraph)，即不能同时被多个线程/进程使用。

---

## Pytorch Dataloader的多线程
Torch 在 dataloader 里面默认的方法是 **Fork**，还可以选择**Spawn**，这两者的区别可见[中文简单解释](https://blog.csdn.net/qq_28327765/article/details/120495877)和[stackoverflow比较详细的解释](https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn)。简而言之，如下:
+ **Fork**：复制父进程的所有资源，包括 socket FD。结果就是多个进程用同一个连接句柄，请求/响应包顺序容易乱掉（第一次我用 gclient 的版本就是这样挂的）。

+ **Spawn**：不复制父进程资源，子进程完全重建连接。这样就没有 FD 冲突，但启动很慢，而且我当时没解决调用 `mp` 报错的问题，所以没用上。

我自己测试 spawn 的调用方法：
``` python
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
```

---

## 工厂 + 懒加载

当前代码中还是`fork`方法。解决`fork`中因为复制造成的相同`socket FD`的问题，使用了工厂函数和懒加载，只有在被调用的时候再进行conn pool和session的加载，即发生在多进程产生之后，这样的话，调用session就不会出现上面fork所说会出现的 request和response不一致的问题。

那么为什么还要引入`pid`呢？第一，`pid`的引入一个是DataLoader的persistent机制，当persistent = false 的时候，DataLoader 的 worker 会被反复拉起/退出，这个时候`pid`会存在变化。其次，增加鲁棒性，相当于给进程的取session上了保险。一个是，如果还有在子进程内还出现了什么进程（一些环境依赖）的fork，那么`pid`能保证不会出现问题；另一个是在未来扩展上，可以相对自由的不用关心多进程会造成这一块出现的问题。
这一部分的开销非常小（是不是可以做一个具体的检测），是一次整数的比较判读。

---

## 性能效果
基于以上，设计了现在的[base_store.py](../nebula_pyg/base_store.py)。

以下提供加入不同数量`number_worker`的在训练[OGBN-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)的速度对比（环境：WSL Ubuntu22.04，CPU: AMD R7 5800H(8c16t)，DDR4 40G，NebulaGraph 3.8.0 Docker-Compos, Torch2.7.1+cpu）：


| num_workers        | 1      | 2      | 4      | 8      | 16     |
| ------------------ | ------ | ------ | ------ | ------ | ------ |
| per Epoch          | 19m20s | 10m21s | 5m51s  | 4m53s  | 4m04s  |
| 5 Epoch            | 96m41s | 51m48s | 29m15s | 24m28s | 20m22s |
| average iterator/s | 4.56   | 8.51   | 15.08  | 18.02  | 21.65  |


可以看到 8 以上的 worker 数加速效果就不太明显了，这可能是因为 CPU 核心数、NebulaGraph 查询吞吐等限制，可以按主机配置调。