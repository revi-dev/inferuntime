import pathlib
import tempfile

import onnx
from onnxsim import simplify
import torch
import torch.nn as nn


def convert_to_onnx_model(
    model: nn.Module,
    input_shape: tuple[int],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] = None,
    opset_version: int = 17
) -> tuple[onnx.ModelProto, bool]:
    model.to('cpu')
    
    input_names = input_names or ['input']
    output_names = output_names or ['output']
    dummy_input = torch.randn(*input_shape, requires_grad=False)
    
    model.eval()
    
    with tempfile.NamedTemporaryFile(mode='w+b') as t:
        torch.onnx.export(model=model,
                          args=dummy_input,
                          f=t.name,
                          export_params=True,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes
                          )

        model = onnx.load(t.name)
        
        # 下記のUserWarningへの対処
        # UserWarning: ONNX export mode is set to TrainingMode.EVAL, but operator 'batch_norm' is set to train=True. Exporting with train=True
        # https://stackoverflow.com/questions/77486728/batchnorms-force-set-to-training-mode-on-torch-onnx-export-when-running-stats-ar
        for node in model.graph.node:
            if node.op_type == "BatchNormalization":
                for attribute in node.attribute:
                    if attribute.name == 'training_mode':
                        if attribute.i == 1:
                            node.output.remove(node.output[1])
                            node.output.remove(node.output[1])
                        attribute.i = 0
        
        simplified_model, check = simplify(model)

    return simplified_model, check


def export_onnx_model(
    model: nn.Module,
    filename: str,
    input_shape: tuple[int],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] = None,
    opset_version: int = 17
) -> bool:
    filepath = pathlib.Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    simplified_model, check \
        = convert_to_onnx_model(model=model,
                                input_shape=input_shape,
                                input_names=input_names,
                                output_names=output_names,
                                dynamic_axes=dynamic_axes,
                                opset_version=opset_version
                                )

    onnx.save(simplified_model, str(filepath))

    return check
