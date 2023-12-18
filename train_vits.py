from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, VitsArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.characters import CharactersConfig
from vn_characters.vn_characters import VieCharacters
from formater.customformater import custom_formatter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', help='output path for the training process', type=str, default=None)
parser.add_argument('--data_path', help='path to the dataset directory', type=str, default=None)
parser.add_argument('--restore_path',
                    help='Path to a model checkpoint. Restore the model with the given checkpoint and start a new training.',
                    type=str,
                    default=None)
parser.add_argument('--epoch', help='number of epoch', type=int, default=2000)
parser.add_argument('--batch_size', help='batch size', type=int, default=64)
parser.add_argument('--lr', help='learning rate', type=float, default=2e-4)
parser.add_argument('--eval_batch_size', help='eval batch size', type=int, default=32)
parser.add_argument('--continue_path', help="Path to a training folder to continue training.", type=str, default=None)
parser.add_argument('--sample_rate', type=int, default=22050)
parser.add_argument('--meta_filename', type=str, help='name of the metadata file')

args = parser.parse_args()

if __name__ == '__main__':
    # Init dataset and audio config
    dataset_config = BaseDatasetConfig(meta_file_train=args.meta_filename, path=args.data_path)
    audio_config = VitsAudioConfig(
        sample_rate=args.sample_rate,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    # Init character and character config
    vie_characters = VieCharacters()
    character_config = CharactersConfig(
        pad=vie_characters.pad,
        eos=vie_characters.eos,
        bos=vie_characters.bos,
        blank=vie_characters.blank,
        punctuations=vie_characters.punctuations,
        characters=vie_characters.characters
    )

    # Init model config
    config = VitsConfig(
        model_args=VitsArgs(num_chars=vie_characters.num_chars),
        audio=audio_config,
        run_name="vits_viettts",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epoch,
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=args.output_path,
        datasets=[dataset_config],
        characters=character_config,
        cudnn_benchmark=False,
        lr_disc=args.lr,
        lr_gen=args.lr
    )

    # Init audio processor
    ap = AudioProcessor.init_from_config(config)

    # Init tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data samples
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=custom_formatter,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer
    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path,
                    restore_path=args.restore_path),
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
