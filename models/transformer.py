import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class TransformerForecasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        h_channels=512,
        out_channels=1,
        dropout=0.1,
        lr=1e-5,
        use_periodic_encoder=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_encoder_inputs = n_encoder_inputs
        self.n_decoder_inputs = n_decoder_inputs
        self.lr = lr
        self.dropout = dropout
        self.use_periodic_encoder=use_periodic_encoder
        
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=h_channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=h_channels)
        
        self.periodic_embedding = torch.nn.Embedding(32, embedding_dim=h_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * h_channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=h_channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * h_channels,
        )

        self.encoder =  torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, h_channels)
        self.output_projection = Linear(n_decoder_inputs, h_channels)

        self.linear = Linear(h_channels, out_channels)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):

        src_start = self.input_projection(src[:, :, :self.n_encoder_inputs]).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )  # pos_encoder: [batch_size, in_sequence_len]
        
        pos_encoder = self.input_pos_embedding(pos_encoder)
        pos_encoder = pos_encoder.permute(1, 0, 2)  # pos_encoder: [in_sequence_len, batch_size, h_channels]
    
        src_out = src_start + pos_encoder
        
        if self.use_periodic_encoder:
            periodic_encoder = src[:, :, -1].long()  # periodic_encoder: [batch_size, in_sequence_len]            
            periodic_encoder = self.periodic_embedding(periodic_encoder)
            periodic_encoder = periodic_encoder.permute(1, 0, 2) # periodic_encoder: [in_sequence_len, batch_size, h_channels]

            src_out = src_out + periodic_encoder
            
        out = self.encoder(src_out) + src_start
        return out

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg[:, :, :self.n_decoder_inputs]).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_encoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )  # pos_encoder: [batch_size, in_sequence_len]
        
        pos_encoder = self.input_pos_embedding(pos_encoder)
        pos_encoder = pos_encoder.permute(1, 0, 2)  # pos_decoder: [in_sequence_len, batch_size, h_channels]
        
        trg_out = trg_start + pos_encoder
        
        if self.use_periodic_encoder:
            periodic_encoder = trg[:, :, -1].long()  # periodic_decoder: [batch_size, in_sequence_len]
            periodic_encoder = self.periodic_embedding(periodic_encoder)
            periodic_encoder = periodic_encoder.permute(1, 0, 2) # periodic_decoder: [in_sequence_len, batch_size, h_channels]
            
            trg_out = trg_out + periodic_encoder
            
        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg_out, memory=memory, tgt_mask=trg_mask) + trg_start
        out = out.permute(1, 0, 2)
        out = self.linear(out)
        return out

    def forward(self, x):
        
        src, trg = x
        
        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    n_classes = 100

    source = torch.rand(size=(32, 16, 9))
    target_in = torch.rand(size=(32, 16, 8))
    target_out = torch.rand(size=(32, 16, 1))

    ts = TimeSeriesForcasting(n_encoder_inputs=9, n_decoder_inputs=8)

    pred = ts((source, target_in))

    print(pred.size())

    ts.training_step((source, target_in, target_out), batch_idx=1)