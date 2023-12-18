from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
import numpy as np
import time
import torch


def load_model_from_config(config_path) -> (Vits, VitsConfig):
    config = VitsConfig()
    config.load_json(config_path)
    vits = Vits.init_from_config(config)
    return vits, config


def inference_onnx(vits: Vits, config: VitsConfig, text: str, model_onnx_path: str, output_file_wav: str) -> None:
    """

    Args:
        vits: VITS model
        config: Vits config
        text: your text
        model_onnx_path: path to the .onnx file
        output_file_wav: path to the .wav file

    Returns:

    """
    text = text.lower()
    vits.load_onnx(model_onnx_path)
    text_inputs = np.asarray(
        vits.tokenizer.text_to_ids(text),
        dtype=np.int64,
    )[None, :]
    start = time.perf_counter()
    audio = vits.inference_onnx(x=text_inputs)
    end = time.perf_counter()
    inference_time = end - start
    audio_length = audio.shape[1] / config.audio.sample_rate
    print('Inference time: {}'.format(inference_time))
    print('Audio length: {}'.format(audio_length))
    print('Real time factor: {}'.format(inference_time / audio_length))
    save_wav(wav=audio[0], path=output_file_wav, sample_rate=config.audio.sample_rate)


def inference(vits: Vits, config: VitsConfig, text: str, model_checkpoint_path: str, output_file_wav: str) -> None:
    """

    Args:
        vits: the VITS model
        config: the VITS config
        text: your text
        model_checkpoint_path: path to the .pth file
        output_file_wav: path to save the output wav file

    Returns: None

    """
    text = text.lower()
    vits.load_checkpoint(config, model_checkpoint_path)
    text_inputs = vits.tokenizer.text_to_ids(text)
    text_inputs = torch.tensor(text_inputs).unsqueeze(0)
    start = time.perf_counter()
    audio = vits.inference(x=text_inputs)['model_outputs']
    audio_length = audio.shape[-1] / config.audio.sample_rate
    end = time.perf_counter()
    inference_time = end - start
    print('Inference time: {}'.format(inference_time))
    print('Audio length: {}'.format(audio_length))
    print('Real time factor: {}'.format(inference_time / audio_length))


if __name__ == '__main__':
    vits, config = load_model_from_config('config.json')
    text = 'Bộ trưởng cho biết đã có 45 chuyến thăm của các lãnh đạo Việt Nam chủ chốt tới các nước láng giềng, các nước đối tác chiến lược.'
    inference_onnx(vits=vits, config=config, text=text, model_onnx_path='coqui_vits.onnx', output_file_wav='out.wav')
    # inference(vits=vits, config=config, text=text, model_checkpoint_path='checkpoint_100000.pth',
    #           output_file_wav='out1.wav')
