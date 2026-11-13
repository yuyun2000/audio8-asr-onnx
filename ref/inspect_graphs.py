#!/usr/bin/env python3
"""Dump ONNX graph IO table + structural facts for the official Audio8-ASR bundle.

Writes ref/onnx_io.json and prints a markdown-ready table.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx

DT = {v: k for k, v in onnx.TensorProto.DataType.items()} if hasattr(onnx.TensorProto, "DataType") else {}
DTYPE_NAME = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
    7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128", 16: "bfloat16",
}


def shape_of(vi) -> list:
    out = []
    t = vi.type.tensor_type
    for d in t.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def io_table(model_path: Path) -> dict:
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    inputs = []
    for vi in g.input:
        if vi.name in inits:
            continue  # initializer, not a real runtime input
        inputs.append({
            "name": vi.name,
            "dtype": DTYPE_NAME.get(vi.type.tensor_type.elem_type, vi.type.tensor_type.elem_type),
            "shape": shape_of(vi),
        })
    outputs = [{
        "name": vi.name,
        "dtype": DTYPE_NAME.get(vi.type.tensor_type.elem_type, vi.type.tensor_type.elem_type),
        "shape": shape_of(vi),
    } for vi in g.output]
    ops = Counter(n.op_type for n in g.node)
    # structural probes
    const_names = {i.name for i in g.initializer}
    init_bytes = 0
    import numpy as np
    for i in g.initializer:
        n = 1
        for d in i.dims:
            n *= d
        init_bytes += n * np.dtype(DTYPE_NAME.get(i.data_type, "float32")).itemsize
    out = {
        "file": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "ir_version": m.ir_version,
        "producer": f"{m.producer_name} {m.producer_version}",
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in m.opset_import],
        "num_nodes": len(g.node),
        "num_initializers": len(g.initializer),
        "initializer_bytes": init_bytes,
        "inputs": inputs,
        "outputs": outputs,
        "op_types": dict(ops.most_common()),
        "has_sin_cos": bool({"Cos", "Sin"} & set(ops)),
        "has_rope_like": bool({"RotaryEmbedding"} & set(ops)) or (
            "Cos" in ops and "Sin" in ops and "Mul" in ops
        ),
        "node_sample": [f"{n.op_type}({','.join(n.input)})->{','.join(n.output)}" for n in g.node[:0]],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="downloaded/onnx_release/model_bundle")
    ap.add_argument("--out", default="ref/onnx_io.json")
    args = ap.parse_args()
    d = Path(args.dir)
    names = ["audio_hidden.onnx", "lm_cache_prefill.onnx", "lm_cache_decode.onnx", "lm_logits.onnx"]
    res = {}
    for n in names:
        p = d / n
        if not p.exists():
            print(f"MISSING {p}")
            continue
        r = io_table(p)
        res[n] = r
        print(f"\n=== {n}  ({r['size_bytes']/1e6:.1f} MB, {r['num_nodes']} nodes, "
              f"{r['num_initializers']} initializers/{r['initializer_bytes']/1e6:.1f} MB, "
              f"opset {r['opset']}, sq_cos={r['has_sin_cos']})")
        print("  IN :")
        for i in r["inputs"]:
            print(f"    {i['name']:24s} {i['dtype']:8s} {i['shape']}")
        print("  OUT:")
        for o in r["outputs"]:
            print(f"    {o['name']:24s} {o['dtype']:8s} {o['shape']}")
        print("  ops:", ", ".join(f"{k}x{v}" for k, v in r["op_types"].items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
