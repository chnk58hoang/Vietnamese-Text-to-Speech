from TTS.tts.utils.text.characters import BaseCharacters

_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # TODO: check if we need this alongside with PAD
_characters = 'abcdefghijklmnopqrstuvwxyzàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
_punctuations = "!'(),-.:;? "


class VieCharacters(BaseCharacters):
    def __init__(self, characters: str = _characters,
                 punctuations: str = _punctuations,
                 pad: str = _pad,
                 eos: str = _eos,
                 bos: str = _bos,
                 blank: str = _blank,
                 is_unique: bool = False,
                 is_sorted: bool = True) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    def _create_vocab(self):
        return super()._create_vocab()
