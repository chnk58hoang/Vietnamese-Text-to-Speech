import os
import string


def custom_formatter(root_path, metafile, **kwargs):
    """Custom formatter for loading data sample

    Args:
        root_path: (str): path to the data directory
        metafile: (str): name of the metadata file

    Returns:
        List of data samples with format:
        {"text": clean_text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path}
    """

    # Get path to the tsv metadata file
    tsv_file = os.path.join(root_path, metafile)

    items = []
    speaker_name = "my_speaker"

    with open(tsv_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("\t")
            if cols[1] != 'wav':
                # Get the path to the wav file
                wav_file = os.path.join(root_path, 'infore_16k_denoised', cols[1])
                # Get the corresponding text and clean it
                text = cols[2][:-1]
                text = text.lower()
                clean_text = text.translate(str.maketrans('', '', string.punctuation))
                # Get all necessary information
                items.append(
                    {"text": clean_text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items
