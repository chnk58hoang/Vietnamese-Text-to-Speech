from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
import numpy as np


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


def inference(model_config_path: str, text: str, model_onnx_path: str, output_file_wav: str) -> None:
    config = VitsConfig()
    config.load_json(model_config_path)
    vits = Vits.init_from_config(config)

    vits.load_onnx(model_onnx_path)
    text_inputs = np.asarray(
        vits.tokenizer.text_to_ids(text),
        dtype=np.int64,
    )[None, :]

    audio = vits.inference_onnx(text_inputs)
    save_wav(wav=audio[0], path=output_file_wav, sample_rate=config.audio.sample_rate)
