# scripts/import_cora_to_nebula.py
import os
import time
from tqdm import tqdm
import torch
from torch_geometric.datasets import Planetoid

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

SPACE = "cora"
USER = "root"
PASSWORD = "nebula"

USE_STRING_VID = True

GRAPH_HOSTS = [("graphd", 9669)]

# Whether to write feature x to Nebula (LIST<DOUBLE>), default is False (lighter)
STORE_X = False


def make_pool():
    cfg = Config()
    cfg.max_retry_connect = 1
    pool = ConnectionPool()
    ok = pool.init(GRAPH_HOSTS, cfg)
    assert ok, "Init ConnectionPool failed"
    return pool


def run_ngql(session, ngql: str):
    resp = session.execute(ngql)
    if not resp.is_succeeded():
        raise RuntimeError(f"NGQL failed: {ngql}\n{resp.error_msg()}")
    return resp


def vid_literal(v: int) -> str:
    return f'"{v}"' if USE_STRING_VID else str(v)


def main():
    root = os.path.join(os.path.dirname(__file__), "../..", "data")
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    N = data.num_nodes
    E = data.edge_index.size(1)
    # print(f"[Cora] num_nodes={N}  num_edges={E}  x={tuple(data.x.size())}  y={tuple(data.y.size())}")

    pool = make_pool()
    session = pool.get_session(USER, PASSWORD)

    # 3) Create space & schema
    # Note: If there are previously different vid_type spaces, you need to DROP SPACE first
    print("[Nebula] Create space & schema...")
    run_ngql(
        session, f"CREATE SPACE IF NOT EXISTS {SPACE}(vid_type = FIXED_STRING(32));"
    )
    time.sleep(6.0)
    run_ngql(session, f"USE {SPACE};")
    # Tag: Paper(label int{, x list<double>})
    if STORE_X:
        run_ngql(session, "CREATE TAG IF NOT EXISTS Paper(label int, x list<double>);")
    else:
        run_ngql(session, "CREATE TAG IF NOT EXISTS Paper(label int);")
    # Edge: Cites()
    run_ngql(session, "CREATE EDGE IF NOT EXISTS Cites();")

    time.sleep(6.0)

    # Inserting vertices
    print("[Nebula] Inserting vertices...")
    BATCH = 2000
    y = data.y.cpu().tolist()

    if STORE_X:
        x = data.x.cpu().tolist()  # [N, 1433] → list[list[float]]

    for i in tqdm(range(0, N, BATCH)):
        vals = []
        for vid in range(i, min(i + BATCH, N)):
            label = int(y[vid])
            if STORE_X:
                # Nebula LIST<DOUBLE>: 用 [a,b,c] 语法
                feat = ",".join(f"{float(v):.6f}" for v in x[vid])
                vals.append(f'"{vid}":( {label}, [{feat}] )')
            else:
                vals.append(f'"{vid}":({label})')
        cols = "label" + (", x" if STORE_X else "")
        stmt = f"INSERT VERTEX Paper({cols}) VALUES " + ", ".join(vals) + ";"
        run_ngql(session, stmt)

    # Inserting edges (both directions)
    print("[Nebula] Inserting edges (both directions)...")
    ei = data.edge_index.cpu()
    # Deduplication: When splitting an undirected edge into two directed edges, first deduplication the original pair.
    edges = set()
    for k in range(E):
        u = int(ei[0, k])
        v = int(ei[1, k])
        if u == v:
            continue
        # Deduplication: When splitting an undirected edge into two directed edges, first deduplication the original pair.
        a, b = (u, v) if u <= v else (v, u)
        edges.add((a, b))

    edges = list(edges)
    E_uniq = len(edges)
    # print(f"[Nebula] unique undirected pairs = {E_uniq}, will insert {E_uniq*2} directed edges")

    for i in tqdm(range(0, E_uniq, BATCH)):
        part = edges[i : i + BATCH]
        vals = []
        for u, v in part:
            vals.append(f"{vid_literal(u)}->{vid_literal(v)}:()")
            vals.append(f"{vid_literal(v)}->{vid_literal(u)}:()")
        stmt = "INSERT EDGE Cites() VALUES " + ", ".join(vals) + ";"
        run_ngql(session, stmt)

    print("[Nebula] Done.")
    session.release()
    pool.close()


if __name__ == "__main__":
    main()
