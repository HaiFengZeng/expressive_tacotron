from train import prepare_dataloaders
from hparams import create_hparams

hparams = create_hparams()


def test_dataloader():
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    for i, batch in enumerate(train_loader):
        print(batch)
        break
