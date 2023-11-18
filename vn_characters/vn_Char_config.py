from TTS.tts.utils.text.characters import CharactersConfig
from vn_characters.vn_characters import VieCharacters



class Vie_CharacterConfig(CharactersConfig):
    
    characters_class = VieCharacters
    characters = VieCharacters.characters
    blank = VieCharacters.blank
    eos = VieCharacters.eos
    bos = VieCharacters.bos
    pad = VieCharacters.pad
    punctuations = VieCharacters.punctuations

    