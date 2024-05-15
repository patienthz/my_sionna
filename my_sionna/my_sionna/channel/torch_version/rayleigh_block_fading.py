import torch
from . import ChannelModel

class RayleighBlockFading_torch(ChannelModel):
    def __init__(self, num_rx, num_rx_ant, num_tx, num_tx_ant):
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.num_rx = num_rx
        self.num_rx_ant = num_rx_ant

    def __call__(self, batch_size, num_time_steps, sampling_frequency=None):
        # Delays
        delays = torch.zeros([batch_size, self.num_rx, self.num_tx, 1])

        # Fading coefficients
        std = torch.sqrt(torch.tensor(0.5))
        h_real = torch.randn([batch_size, self.num_rx, self.num_rx_ant, self.num_tx, self.num_tx_ant, 1, 1]) * std
        h_img = torch.randn([batch_size, self.num_rx, self.num_rx_ant, self.num_tx, self.num_tx_ant, 1, 1]) * std
        h = torch.complex(h_real, h_img)

        # Tile the response over the block
        h = h.repeat(1, 1, 1, 1, 1, 1, num_time_steps)
        return h, delays