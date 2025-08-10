from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.data.DataObject import ValueWrapper
from typing import Dict, List, Tuple, Callable, Optional, Iterable

import torch


# TODO: This is used in the get_tensor method of the old version of NebulaFeatureStore and can be deleted.
def split_batches(lst, batch_size):
    """
    Generator that splits a list into smaller batches.

    Args:
        lst (list): The list to split.
        batch_size (int): The maximum size of each batch.

    Yields:
        list: Sub-list of size <= batch_size.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def scan_all_tag_vids(
        space: str,
        session,
        sclient,
        batch_size: int = 4096,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[int, str]], Dict[str, str]]:
    """
    Scans all vertex tags in the given Nebula space and collects all VIDs.

    Args:
        space (str): The name of the Nebula space.
        gclient: NebulaGraph graph client.
        sclient: NebulaGraph storage client.
        batch_size (int): Number of vertices to scan per batch.

    Returns:
        dict: Mapping from tag name to list of vertex IDs.
              Format: { tag: [vid1, vid2, ...] }
    """
    session.execute(f"USE {space};")
    tags_result = session.execute("SHOW TAGS;")
    tags = [ValueWrapper(row.values[0]).cast() for row in tags_result.rows()]

    vid_to_idx: Dict[str, Dict[str, int]] = {t: {} for t in tags}
    idx_to_vid: Dict[str, Dict[int, str]] = {t: {} for t in tags}
    vid_to_tag: Dict[str, str] = {}

    for tag in tags:
        i = 0
        for _part_id, batch in sclient.scan_vertex_async(space, tag, prop_names=[], batch_size=batch_size):
            for node in batch.as_nodes():
                vid = node.get_id().cast()
                vid_to_idx[tag][vid] = i
                idx_to_vid[tag][i] = vid
                vid_to_tag[vid] = tag
                i += 1

    return vid_to_idx, idx_to_vid, vid_to_tag


def get_edge_type_groups(
    session,                       # ← graphd session（已登录）
    sclient,                       # ← GraphStorageClient
    space: str,
    snapshot: Dict,                # 需要包含 'vid_to_tag'
    batch_size: int = 4096,
    sample_per_edge: int = 1,      # 每种 edge_name 抽样多少条边来推断三元组
    edge_names: Optional[Iterable[str]] = None,  # 如已知可直接传，省一次 SHOW
) -> List[Tuple[str, str, str]]:
    """
    Infers actual (src_tag, edge_type, dst_tag) triples present in the graph.

    This uses the snapshot["vid_to_tag"] to determine the tag types of each VID.

    Args:
        gclient: NebulaGraph graph client.
        sclient: NebulaGraph storage client.
        space (str): Nebula space name.
        snapshot (dict): Must include 'vid_to_tag': { vid: tag }.

    Returns:
        set: A set of 3-tuples (src_tag, edge_type, dst_tag).
    """
    session.execute(f"USE {space};")
    if edge_names is None:
        rs = session.execute("SHOW EDGES;")
        edge_names = [ValueWrapper(r.values[0]).cast() for r in rs.rows()]

    vid_to_tag: Dict[str, str] = snapshot.get("vid_to_tag", {})
    groups = set()

    for edge_name in edge_names:
        found = 0
        for _part_id, batch in sclient.scan_edge_async(
            space,
            edge_name,
            prop_names=[],
            batch_size=batch_size,
        ):
            for rel in batch.as_relationships():
                svid = rel.start_vertex_id().cast()
                dvid = rel.end_vertex_id().cast()
                src_tag = vid_to_tag.get(svid)
                dst_tag = vid_to_tag.get(dvid)
                if src_tag is not None and dst_tag is not None:
                    groups.add((src_tag, edge_name, dst_tag))
                    found += 1
                    if found >= sample_per_edge:
                        break
            if found >= sample_per_edge:
                break

    return sorted(groups)


def build_edge_index_dict(gclient, sclient, space, snapshot):
    """
    Builds a PyG-compatible edge_index_dict using NebulaGraph data.

    The result maps each (src_tag, edge_type, dst_tag) group to a COO edge index tensor.

    Args:
        space (str): The Nebula space name.
        sclient: NebulaGraph storage client.
        gclient: NebulaGraph query client.
        snapshot (dict): Must include:
            - vid_to_idx: { tag: {vid: index} }
            - vid_to_tag: { vid: tag }

    Returns:
        dict: PyG HeteroData-style edge_index dictionary:
              { (src_tag, edge_type, dst_tag): torch.LongTensor([2, num_edges]) }
    """
    edge_index_dict = {}
    edge_types = [ValueWrapper(row.values[0]).cast() for row in gclient.execute(f"USE {space}; SHOW EDGES;").rows()]
    for edge_type in edge_types:
        # Group edges by (src_tag, dst_tag)
        buf = {}
        for part_id, batch in sclient.scan_edge_async(space, edge_type, batch_size=4096):
            for rel in batch.as_relationships():
                src_vid = rel.start_vertex_id().cast()
                dst_vid = rel.end_vertex_id().cast()

                src_tag = snapshot["vid_to_tag"].get(src_vid, None)
                dst_tag = snapshot["vid_to_tag"].get(dst_vid, None)
                if src_tag is None or dst_tag is None:
                    continue

                key = (src_tag, edge_type, dst_tag)
                if key not in buf:
                    buf[key] = ([], [])

                src_idx = snapshot["vid_to_idx"][src_tag][src_vid]
                dst_idx = snapshot["vid_to_idx"][dst_tag][dst_vid]
                buf[key][0].append(src_idx)
                buf[key][1].append(dst_idx)

        for key in buf:
            edge_index = torch.tensor(buf[key], dtype=torch.long)
            edge_index_dict[key] = edge_index

    return edge_index_dict
