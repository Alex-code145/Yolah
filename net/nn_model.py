import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple convolutional network producing a policy over 64x64 move pairs and a scalar value.
# Input: tensor (batch, 3, 8, 8) -> channels: black, white, empty
# Output: policy logits (batch, 4096), value (batch, 1)

OUTPUT_POLICY_SIZE = 64 * 64

class YolahNet(nn.Module):
    def __init__(self, in_channels=3, channels=64, policy_size=OUTPUT_POLICY_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 3, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)


# Helper: convert a YolahState into tensor input expected by the network
# We avoid importing the game module here to keep the model decoupled. The caller should provide
# integers black, white, empty and current_player when calling state_to_tensor.

def state_to_tensor(black: int, white: int, empty: int, current_player: int):
    # produce tensor shape (3, 8, 8) as floats
    def bits_to_plane(bb: int):
        plane = torch.zeros((8, 8), dtype=torch.float32)
        n = 0
        while bb:
            if bb & 1:
                i = n // 8
                j = n % 8
                plane[i, j] = 1.0
            bb >>= 1
            n += 1
        return plane

    b_plane = bits_to_plane(black)
    w_plane = bits_to_plane(white)
    e_plane = bits_to_plane(empty)
    # add current player as a constant plane (optional)
    cp_plane = torch.full((8, 8), float(current_player), dtype=torch.float32)
    # Stack: black, white, empty (we keep 3 channels to match constructor), ignore cp for now
    inp = torch.stack([b_plane, w_plane, e_plane], dim=0)
    return inp


# Policy / move utilities

def move_to_index(move):
    # move is a Move-like object with from_sq.sq and to_sq.sq attributes (0..63)
    return move.from_sq.sq * 64 + move.to_sq.sq


def index_to_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return from_sq, to_sq
