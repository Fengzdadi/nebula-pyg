from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.sclient.GraphStorageClient import GraphStorageClient
from nebula3.data.DataObject import ValueWrapper

import torch

def split_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def scan_all_tag_vids(space, gclient, sclient, batch_size=4096):
    """
    扫描所有 tag，返回 dict: {tag: [vid1, vid2, ...]}
    """
    gclient.execute(f"USE {space};")
    tags_result = gclient.execute("SHOW TAGS;")
    tags = [ValueWrapper(row.values[0]).cast() for row in tags_result.rows()]
    tag_vids = {tag: [] for tag in tags}
    for tag in tags:
        for part_id, batch in sclient.scan_vertex_async(space, tag, prop_names=[], batch_size=batch_size):
            for node in batch.as_nodes():
                vid = node.get_id().cast()
                tag_vids[tag].append(vid)
    return tag_vids  # dict: {tag: [vid1, vid2, ...]}

def get_edge_type_groups(space, sclient, gclient, snapshot):
    """
    返回实际存在的所有 (src_tag, edge_type, dst_tag) 分组集合
    snapshot 需有 vid_to_tag 字典: {vid: tag}
    """
    groups = set()
    # SHOW EDGES 获取所有 edge_type
    edge_types = [ValueWrapper(row.values[0]).cast() for row in gclient.execute(f"USE {space}; SHOW EDGES;").rows()]
    for edge_type in edge_types:
        for part_id, batch in sclient.scan_edge_async(space, edge_type, batch_size=4096):
            for rel in batch.as_relationships():
                src_vid = rel.start_vertex_id().cast()
                dst_vid = rel.end_vertex_id().cast()
                src_tag = snapshot["vid_to_tag"].get(src_vid, None)
                dst_tag = snapshot["vid_to_tag"].get(dst_vid, None)
                if src_tag is not None and dst_tag is not None:
                    groups.add((src_tag, edge_type, dst_tag))
    return groups

def build_edge_index_dict(space, sclient, gclient, snapshot):
    """
    返回 PyG 标准 HeteroData edge_index_dict:
    { (src_tag, edge_type, dst_tag): edge_index_tensor }
    snapshot: 必须包含 vid_to_idx, vid_to_tag
    """
    edge_index_dict = {}
    edge_types = [ValueWrapper(row.values[0]).cast() for row in gclient.execute(f"USE {space}; SHOW EDGES;").rows()]
    for edge_type in edge_types:
        # 先自动分组所有 (src_tag, dst_tag)
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
                # 用 per-tag 的编号
                src_idx = snapshot["vid_to_idx"][src_tag][src_vid]
                dst_idx = snapshot["vid_to_idx"][dst_tag][dst_vid]
                buf[key][0].append(src_idx)
                buf[key][1].append(dst_idx)
        for key in buf:
            edge_index = torch.tensor(buf[key], dtype=torch.long)
            edge_index_dict[key] = edge_index
    return edge_index_dict

