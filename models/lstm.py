import random

import torch
import torch.nn as nn
from torch.nn import Linear
import pytorch_lightning as pl


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


class LSTMEncoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.linear = nn.Linear(1, 1)       
        
    def forward(self, x, hidden_state):  # Inputs: input, (h_0, c_0)
        
        '''
        : param x_input:               input of shape [seq_len, batch_size, input_size]
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        
        lstm_out, self.hidden = self.lstm(x, hidden_state)  # lstm_out: [seq_len, batch_size, input_size]
        return lstm_out, self.hidden  # Outputs: output, (h_n, c_n)
    
    def init_hidden(self, batch_size):
        
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                            device=self.linear.weight.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size,
                            device=self.linear.weight.device))
    

class LSTMDecoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        
        self.linear = nn.Linear(hidden_size, output_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))     
        
        return output, self.hidden

    
class LSTMForecasting(pl.LightningModule):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        h_channels=512,
        out_channels=1,
        lr=1e-5,
        dropout=0.1,
        teacher_forcing_ratio=1.
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_encoder_inputs = n_encoder_inputs
        self.n_decoder_inputs = n_decoder_inputs        
        self.lr = lr
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = LSTMEncoder(input_size=n_encoder_inputs, 
                                   hidden_size=h_channels)
        
        self.decoder = LSTMDecoder(input_size=n_decoder_inputs, 
                                   hidden_size=h_channels, 
                                   output_size=out_channels)

        self.do = nn.Dropout(p=self.dropout)

    def forward(self, x, teacher_forcing_ratio=1):
        
        src, trg = x
        src = src[:, :, :self.n_encoder_inputs] 
        trg = trg[:, :, :self.n_decoder_inputs]
        
        batch_size, seq_len, _ = src.shape
        _, horizon, _ = trg.shape
        
        outputs = [] #torch.zeros(horizon, batch_size, trg.shape[2])
        en_hidden = self.encoder.init_hidden(batch_size)
                
        en_output, en_hidden = self.encoder(x=src, hidden_state=en_hidden)

        # decoder with teacher forcing
        de_hidden = en_hidden
        de_output = trg[:, 0, :]  # shape: (batch_size, input_size)
        
        # use teacher forcing
        if random.random() < self.teacher_forcing_ratio:
            for t in range(horizon): 
                de_input = trg[:, t, :]
                de_output, de_hidden = self.decoder(de_input, de_hidden)
                outputs.append(de_output)
        else:
            for t in range(horizon): 
                de_input = de_output
                de_output, de_hidden = self.decoder(de_input, de_hidden)
                outputs.append(de_output)
                
        outputs = torch.stack(outputs, dim=1)  # shape: [batch_size, horizon, input_size]
        return outputs

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in), teacher_forcing_ratio=self.teacher_forcing_ratio)  # these is an out_channels dimension in the end

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)
        
        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in), teacher_forcing_ratio=0)

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in), teacher_forcing_ratio=0)

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