from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_onnx(model_config_path: str, model_checkpoint_path: str, output_path: str) -> None:
    """Convert Vits model to .onnx
    Args:
        model_config_path: path to the model config.json file
        model_checkpoint_path: path to the model .pth file
        output_path: path to the save .onnx file

    Returns: None
    """
    model_config = VitsConfig()
    model_config.load_json(model_config_path)
    model = Vits.init_from_config(model_config)
    model.load_checkpoint(config=model_config, checkpoint_path=model_checkpoint_path)
    model.export_onnx(output_path=output_path)


def quantize(onnx_path: str, quantized_path: str) -> None:
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)


quantize(onnx_path='coqui_vits.onnx', quantized_path='vits_quantized.onnx')
