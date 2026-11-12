"""Load the local avatar and its animation without modifying the assets."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from gltf._converter import Converter
from gltf.parseutils import parse_gltf_file
from panda3d.core import NodePath


def _append_buffer(data, payload):
    buffer = data["buffers"][0]
    raw = buffer["_glb_bin"]
    raw += b"\0" * (-len(raw) % 4)
    offset = len(raw)
    buffer["_glb_bin"] = raw + payload
    buffer["byteLength"] = len(buffer["_glb_bin"])
    return offset


def _normalize_skin_weights(data):
    """panda3d-gltf 1.3 does not normalize integer WEIGHTS accessors."""
    converted = set()
    for mesh in data.get("meshes", []):
        for primitive in mesh["primitives"]:
            index = primitive.get("attributes", {}).get("WEIGHTS_0")
            if index is None or index in converted:
                continue
            converted.add(index)
            accessor = data["accessors"][index]
            dtype = {5121: np.uint8, 5123: np.dtype("<u2"),
                     5126: np.dtype("<f4")}.get(accessor["componentType"])
            if dtype is None or accessor["type"] != "VEC4":
                raise ValueError("Unsupported skin weight format in avatar GLB.")
            view = data["bufferViews"][accessor["bufferView"]]
            if view["buffer"] != 0 or "sparse" in accessor:
                raise ValueError("Avatar requires a single, non-sparse GLB buffer.")
            itemsize = np.dtype(dtype).itemsize
            weights = np.ndarray(
                (accessor["count"], 4), dtype=dtype,
                buffer=data["buffers"][0]["_glb_bin"],
                offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
                strides=(view.get("byteStride", 4 * itemsize), itemsize),
            ).astype(np.float32)
            sums = weights.sum(axis=1, keepdims=True)
            if not np.isfinite(weights).all() or np.any(sums <= 0):
                raise ValueError("Avatar has invalid skin weights.")
            weights /= sums
            offset = _append_buffer(data, weights.astype("<f4").tobytes())
            accessor.update(componentType=5126, byteOffset=0,
                            bufferView=len(data["bufferViews"]))
            accessor.pop("normalized", None)
            accessor.pop("min", None)
            accessor.pop("max", None)
            data["bufferViews"].append({
                "buffer": 0, "byteOffset": offset, "byteLength": weights.nbytes,
            })


def _attach_idle(data, idle):
    """Merge animation buffers, remapping channel targets by joint name."""
    node_map = {node.get("name"): i for i, node in enumerate(data["nodes"])
                if node.get("name")}
    target_map = {i: node_map[node["name"]]
                  for i, node in enumerate(idle["nodes"])
                  if node.get("name") in node_map}
    offset = _append_buffer(data, idle["buffers"][0]["_glb_bin"])
    view_offset, accessor_offset = len(data["bufferViews"]), len(data["accessors"])
    for source in idle["bufferViews"]:
        view = deepcopy(source)
        if view["buffer"] != 0:
            raise ValueError("Idle animation must use one GLB buffer.")
        view["byteOffset"] = view.get("byteOffset", 0) + offset
        data["bufferViews"].append(view)
    for source in idle["accessors"]:
        accessor = deepcopy(source)
        accessor["bufferView"] += view_offset
        data["accessors"].append(accessor)
    for source in idle.get("animations", []):
        animation = deepcopy(source)
        animation["channels"] = [
            dict(channel, target=dict(channel["target"],
                                     node=target_map[channel["target"]["node"]]))
            for channel in animation["channels"]
            if channel["target"]["node"] in target_map
        ]
        for sampler in animation["samplers"]:
            sampler["input"] += accessor_offset
            sampler["output"] += accessor_offset
        data.setdefault("animations", []).append(animation)


def load_avatar(model_path, idle_path):
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Avatar model missing: {model_path}")
    data = parse_gltf_file(str(model_path))
    # The RPM export names the mesh as its skeleton root. The loader stops at
    # mesh nodes, so start at the actual armature to retain all 68 bones.
    for skin in data.get("skins", []):
        root = skin.get("skeleton")
        if root is not None and "mesh" in data["nodes"][root]:
            roots = [i for i in data["nodes"][root].get("children", [])
                     if "mesh" not in data["nodes"][i]]
            if len(roots) != 1:
                raise ValueError("Cannot identify the avatar's skeleton root.")
            skin["skeleton"] = roots[0]
    _normalize_skin_weights(data)
    if idle_path and Path(idle_path).is_file():
        _attach_idle(data, parse_gltf_file(str(Path(idle_path).resolve())))
    converter = Converter(str(model_path))
    converter.update(data)
    return NodePath(converter.active_scene.node()), [
        name for mesh in data.get("meshes", [])
        for name in mesh.get("extras", {}).get("targetNames", [])
    ]
