#!/usr/bin/env python3
"""Convert a Fast FoundationStereo ONNX graph to FP16.

TensorRT 11 uses strongly typed networks and no longer accepts the legacy
``trtexec --fp16`` switch.  Converting the graph first lets TensorRT build a
real half-precision engine while retaining the same input/output names.
"""

import argparse

import numpy as np
import onnx
from onnxconverter_common import float16
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper


def _repair_half_weight_casts(model):
    """Make explicit float casts feeding half-weight layers consistent.

    Fast FoundationStereo contains a few intentional ``.float()`` operations
    in positional-embedding code.  The generic ONNX converter preserves those
    Cast nodes while converting their convolution weights to half, which is a
    mixed-type graph TensorRT rejects.  A half-weight convolution/matmul can
    safely consume the already-converted half tensor here.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            list(inferred.graph.value_info)
            + list(inferred.graph.input)
            + list(inferred.graph.output)
        )
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.HasField("elem_type")
    }
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    def value_type(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen = seen | {name}
        initializer = initializers.get(name)
        if initializer is not None:
            return initializer.data_type
        producer = producers.get(name)
        if producer is not None and producer.op_type == "Constant":
            tensor_attr = next(
                (attribute for attribute in producer.attribute if attribute.name == "value"),
                None,
            )
            if tensor_attr is not None:
                return tensor_attr.t.data_type
        if producer is not None:
            if producer.op_type == "Cast":
                for attribute in producer.attribute:
                    if attribute.name == "to":
                        return attribute.i
            for input_name in producer.input:
                if input_name:
                    resolved_type = value_type(input_name, seen)
                    if resolved_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
                        return resolved_type
        inferred_type = value_types.get(name)
        if inferred_type is not None:
            return inferred_type
        return None

    repaired = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose", "Gemm", "MatMul") or len(node.input) < 2:
            new_nodes.append(node)
            continue
        weight = initializers.get(node.input[1])
        cast = producers.get(node.input[0])
        if weight is None or weight.data_type != TensorProto.FLOAT16:
            new_nodes.append(node)
            continue
        if cast is not None and cast.op_type == "Cast":
            for attribute in cast.attribute:
                if attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                    attribute.i = TensorProto.FLOAT16
                    repaired += 1
        elif value_type(node.input[0]) == TensorProto.FLOAT:
            original_input = node.input[0]
            cast_output = f"{original_input}__trt11_half_{repaired}"
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [original_input],
                    [cast_output],
                    name=f"{node.name}/TRT11HalfInput",
                    to=TensorProto.FLOAT16,
                )
            )
            node.input[0] = cast_output
            repaired += 1
        new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return repaired


def _repair_mixed_precision_constants(model):
    """Match scalar Constant nodes to the tensor type of their consumer."""
    inferred = onnx.shape_inference.infer_shapes(model)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            list(inferred.graph.value_info)
            + list(inferred.graph.input)
            + list(inferred.graph.output)
        )
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.HasField("elem_type")
    }
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    def value_type(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen = seen | {name}
        initializer = initializers.get(name)
        if initializer is not None:
            return initializer.data_type
        producer = producers.get(name)
        if producer is not None and producer.op_type == "Constant":
            tensor_attr = next(
                (attribute for attribute in producer.attribute if attribute.name == "value"),
                None,
            )
            if tensor_attr is not None:
                return tensor_attr.t.data_type
        if producer is not None:
            if producer.op_type == "Cast":
                for attribute in producer.attribute:
                    if attribute.name == "to":
                        return attribute.i
            for input_name in producer.input:
                if input_name:
                    resolved_type = value_type(input_name, seen)
                    if resolved_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
                        return resolved_type
        inferred_type = value_types.get(name)
        if inferred_type is not None:
            return inferred_type
        return None

    same_type_ops = {
        "Add", "Sub", "Mul", "Div", "Max", "Min", "Pow", "Clip",
        "Greater", "Less", "Equal", "Concat", "GridSample", "MatMul", "Gemm",
    }
    repaired = 0
    new_nodes = []
    for node in model.graph.node:
        if node.op_type not in same_type_ops or not node.input:
            new_nodes.append(node)
            continue
        target_type = value_type(node.input[0])
        if target_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            new_nodes.append(node)
            continue
        # TensorRT requires both matrix-multiply operands to have the same
        # type.  Prefer the weight/right-hand operand's type, since converting
        # an activation to the weight precision is what the FP16 converter
        # intended for these layers.
        if node.op_type in ("MatMul", "Gemm") and len(node.input) > 1:
            rhs_type = value_type(node.input[1])
            if rhs_type in (TensorProto.FLOAT, TensorProto.FLOAT16) and rhs_type != target_type:
                original_input = node.input[0]
                cast_output = f"{original_input}__trt11_matmul_{repaired}"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [original_input],
                        [cast_output],
                        name=f"{node.name}/TRT11MatMulInput",
                        to=rhs_type,
                    )
                )
                node.input[0] = cast_output
                target_type = rhs_type
                repaired += 1
        for input_index, input_name in enumerate(node.input[1:], start=1):
            if not input_name:
                continue
            source_type = value_type(input_name)
            if source_type == target_type:
                continue
            constant = producers.get(input_name)
            if constant is None or constant.op_type != "Constant":
                if source_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
                    continue
                cast_output = f"{input_name}__trt11_same_type_{repaired}"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        [input_name],
                        [cast_output],
                        name=f"{node.name}/TRT11SameType_{input_index}",
                        to=target_type,
                    )
                )
                node.input[input_index] = cast_output
                repaired += 1
                continue
            tensor_attr = next(
                (attribute for attribute in constant.attribute if attribute.name == "value"),
                None,
            )
            if tensor_attr is None or tensor_attr.t.data_type == target_type:
                continue
            if tensor_attr.t.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
                continue
            values = numpy_helper.to_array(tensor_attr.t).astype(
                np.float32 if target_type == TensorProto.FLOAT else np.float16
            )
            tensor_attr.t.CopyFrom(
                numpy_helper.from_array(values, name=tensor_attr.t.name)
            )
            repaired += 1
        new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return repaired


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX float graph to FP16")
    parser.add_argument("--input", required=True, help="Input float32 ONNX file")
    parser.add_argument("--output", required=True, help="Output float16 ONNX file")
    parser.add_argument(
        "--keep_io_types",
        action="store_true",
        help="Keep graph inputs/outputs as float32 and cast internally",
    )
    parser.add_argument(
        "--block_ops",
        default="Clip,Concat,Add",
        help="Comma-separated ops kept in float32 for TensorRT type compatibility",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=args.keep_io_types,
        disable_shape_infer=True,
        op_block_list=[op for op in args.block_ops.split(",") if op],
    )
    repaired_casts = _repair_half_weight_casts(model_fp16)
    repaired_constants = _repair_mixed_precision_constants(model_fp16)
    onnx.save(model_fp16, args.output)
    print(
        f"FP16 ONNX saved to {args.output} "
        f"(repaired casts: {repaired_casts}, constants: {repaired_constants})"
    )


if __name__ == "__main__":
    main()
