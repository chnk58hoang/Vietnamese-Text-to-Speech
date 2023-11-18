import os
import string


        

def custom_formatter(root_path,metafile,**kwargs):
    tsv_file = os.path.join(root_path, metafile)
    items = []
    speaker_name = "my_speaker"

    with open(tsv_file,"r",encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("\t")
            if cols[1] != 'wav':
                wav_file = os.path.join(root_path,'infore_16k_denoised',cols[1])
                text = cols[2][:-1]
                text = text.lower()
                clean_text = text.translate(str.maketrans('','',string.punctuation))
                items.append({"text":clean_text,"audio_file":wav_file,"speaker_name":speaker_name,"root_path":root_path})
    
    return items
