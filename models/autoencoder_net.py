import torch
import torch.nn as nn

class TemporalLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, latent_dim=8, num_layers=2):
        super(TemporalLSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: Transformă secvența de pachete (seq_len, input_dim) într-un vector latent temporal
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Reconstruiește secvența folosind vectorul latent distribuit pe axa timpului
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

    def forward(self, x):
        # x are forma (batch_size, seq_len, input_dim)
        
        # --- ENCODER ---
        # Trecem secvența prin LSTM
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        # Luăm doar starea ascunsă de la ultimul pas de timp pentru a reprezenta secvența comprimată
        last_hidden_state = enc_out[:, -1, :] 
        latent = torch.relu(self.encoder_linear(last_hidden_state))
        
        # --- DECODER ---
        # Recreăm secvența temporală prin repetarea vectorului latent pentru fiecare pas de timp
        seq_len = x.size(1)
        latent_repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_in = torch.relu(self.decoder_linear(latent_repeated))
        dec_out, _ = self.decoder_lstm(dec_in)
        
        return dec_out