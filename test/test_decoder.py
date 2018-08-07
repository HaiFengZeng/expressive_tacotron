from model import Decoder
from hparams import create_hparams

hparams = create_hparams()

def test_decoder():
    decoder = Decoder(hparams)


