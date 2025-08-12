# Multi-process description

The reason for implementing [base\_store.py](../nebula_pyg/base_store.py) came from an issue I encountered while using `FeatureStorage.get_tensor()` with **queries**.
In my case, I needed to fetch properties for a given list of indices from NebulaGraph using a query. (TODO: could write a dedicated note about this.)

Just to insert some background here: PyG’s DataLoader supports the `num_workers` setting, inherited from [PyTorch](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading).
In the original implementation of `get_tensor()`, the user would initialize a `gclient` (session connected to graphd) and an `sclient` (session connected to storaged), and all subsequent operations would go through these same objects.

Once queries were introduced, session operations became extremely frequent (which is actually expected).
Unfortunately, there’s no official documentation, but from various blog posts we can understand PyTorch’s multi-process DataLoader roughly as follows:
The main process maintains an **index queue**, worker processes fetch from it, call `get_tensor()`, and put the results into a **worker\_result\_queue** for the main process to consume.

The problem here is that in a multi-process setting, the `session` used in `get_tensor()` was originally the same object shared across processes, and [NebulaGraph sessions are not thread/process-safe](https://www.nebula-graph.com.cn/posts/informal-analysis-of-session-in-nebulagraph), meaning they cannot be used concurrently across multiple threads or processes.

---

## Multi-process in PyTorch DataLoader

By default, PyTorch DataLoader uses **fork**, but it can also be configured to use **spawn**.
For the differences, see [this Chinese explanation](https://blog.csdn.net/qq_28327765/article/details/120495877) and [a more detailed StackOverflow answer](https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn).
In short:

* **fork**: Copies all resources from the parent process, including socket file descriptors (FDs).
  This means multiple processes end up using the same connection handle, and request/response packets can easily get mixed up (this is exactly what broke my first gclient-based implementation).

* **spawn**: Does not copy parent resources; the child process rebuilds everything from scratch.
  This avoids FD conflicts but is much slower to start, and in my case I didn’t resolve the `mp` call errors at the time, so I couldn’t use it.

Example of how I tested spawn:

```python
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
```

---

## Factory + Lazy Loading

Currently, the code still uses the **fork** method.
To solve the problem of duplicated socket FDs caused by fork, I introduced **factory functions** and **lazy loading**:
The connection pool and session are only created **when first accessed**, which happens **after** worker processes have been spawned.
This ensures each process gets its own session and avoids the request/response mismatch problem seen with fork.

---

## Why introduce `pid` checking?

Two reasons:

1. When `persistent_workers=False`, DataLoader workers are destroyed/recreated between epochs, so their `pid` changes. Without checking, a session might still reference a stale connection.
2. Robustness: if any other processes are forked within a worker (due to environment dependencies), `pid` checking ensures the session always belongs to the current process.
   This also provides flexibility for future extensions without worrying about multi-process safety.
   The overhead is minimal—just a single integer comparison.

---

## Performance Results

Based on the above, I designed the current [base\_store.py](../nebula_pyg/base_store.py).
Here are performance results for different `num_workers` when training on [OGBN-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
(Environment: WSL Ubuntu22.04, CPU: AMD R7 5800H (8c16t), DDR4 40G, NebulaGraph 3.8.0 Docker-Compose, Torch 2.7.1+cpu):

| num\_workers       | 1      | 2      | 4      | 8      | 16     |
| ------------------ | ------ | ------ | ------ | ------ | ------ |
| per Epoch          | 19m20s | 10m21s | 5m51s  | 4m53s  | 4m04s  |
| 5 Epoch            | 96m41s | 51m48s | 29m15s | 24m28s | 20m22s |
| average iterator/s | 4.56   | 8.51   | 15.08  | 18.02  | 21.65  |

You can see that beyond 8 workers, the speedup becomes less significant—likely due to CPU core count and NebulaGraph query throughput limits. Adjust according to your machine’s configuration.
