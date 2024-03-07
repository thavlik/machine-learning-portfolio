# Thank you funkey and timothygebhard for this code.
# Source: https://github.com/timothygebhard/pytorch-conv4d/blob/master/conv4d.py
from __future__ import division

import torch
from torch import nn
from typing import Callable, Tuple, Union

import numpy as np


class Conv4d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int, int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):

        super(Conv4d, self).__init__()

        if type(kernel_size) == int:
            kernel_size = [kernel_size] * 4

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           padding=self.padding)

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv3d_layer.weight)
            if self.bias_initializer is not None:
                self.bias_initializer(conv3d_layer.bias)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        (l_o, d_o, h_o, w_o) = (l_i + 2 * self.padding - l_k + 1,
                                d_i + 2 * self.padding - d_k + 1,
                                h_i + 2 * self.padding - h_k + 1,
                                w_i + 2 * self.padding - w_k + 1)

        # Output tensors for each 3D frame
        frame_results = l_o * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):

            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                frame_conv3d = \
                    self.conv3d_layers[i](input[:, :, j, :]
                                          .view(b, c_i, d_i, h_i, w_i))

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        return torch.stack(frame_results, dim=2)


if __name__ == "__main__":

    print()
    print('TEST PYTORCH CONV4D LAYER IMPLEMENTATION')
    print('\n' + 80 * '-' + '\n')

    # -------------------------------------------------------------------------
    # Generate random input 4D tensor (+ batch dimension, + channel dimension)
    # -------------------------------------------------------------------------

    np.random.seed(42)

    input_numpy = np.round(np.random.random((1, 1, 10, 11, 12, 13)) * 100)
    input_torch = torch.from_numpy(input_numpy).float()

    # -------------------------------------------------------------------------
    # Convolve with a randomly initialized kernel
    # -------------------------------------------------------------------------

    print('Randomly Initialized Kernels:\n')

    # Initialize the 4D convolutional layer with random kernels
    conv4d_layer = \
        Conv4d(in_channels=1,
               out_channels=1,
               kernel_size=(3, 3, 3, 3),
               bias_initializer=lambda x: torch.nn.init.constant_(x, 0))

    # Pass the input tensor through that layer
    output = conv4d_layer.forward(input_torch).data.numpy()

    # Select the 3D kernels for the manual computation and comparison
    kernels = [
        conv4d_layer.conv3d_layers[i].weight.data.numpy().flatten()
        for i in range(3)
    ]

    # Compare the conv4d_layer result and the manual convolution computation
    # at 3 randomly chosen locations
    for i in range(3):

        # Randomly choose a location and select the conv4d_layer output
        loc = [
            np.random.randint(0, output.shape[2] - 2),
            np.random.randint(0, output.shape[3] - 2),
            np.random.randint(0, output.shape[4] - 2),
            np.random.randint(0, output.shape[5] - 2)
        ]
        conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

        # Select slices from the input tensor and compute manual convolution
        slices = [
            input_numpy[0, 0, loc[0] + j, loc[1]:loc[1] + 3, loc[2]:loc[2] + 3,
                        loc[3]:loc[3] + 3].flatten() for j in range(3)
        ]
        manual = np.sum([slices[j] * kernels[j] for j in range(3)])

        # Print comparison
        print(f'At {tuple(loc)}:')
        print(f'\tconv4d:\t{conv4d}')
        print(f'\tmanual:\t{manual}')

    print('\n' + 80 * '-' + '\n')

    # -------------------------------------------------------------------------
    # Convolve with a kernel initialized to be all ones
    # -------------------------------------------------------------------------

    print('Constant Kernels (all 1):\n')

    conv4d_layer = \
        Conv4d(in_channels=1,
               out_channels=1,
               kernel_size=(3, 3, 3, 3),
               padding=1,
               kernel_initializer=lambda x: torch.nn.init.constant_(x, 1),
               bias_initializer=lambda x: torch.nn.init.constant_(x, 0))
    output = conv4d_layer.forward(input_torch)

    # Define relu(x) = max(x, 0) for simplified indexing below
    def relu(x: float) -> float:
        return x * (x > 0)

    # Compare the conv4d_layer result and the manual convolution computation
    # at 3 randomly chosen locations
    for i in range(3):

        # Randomly choose a location and select the conv4d_layer output
        loc = [
            np.random.randint(0, output.shape[2] - 2),
            np.random.randint(0, output.shape[3] - 2),
            np.random.randint(0, output.shape[4] - 2),
            np.random.randint(0, output.shape[5] - 2)
        ]
        conv4d = output[0, 0, loc[0], loc[1], loc[2], loc[3]]

        # For a kernel that is all 1s, we only need to sum up the elements of
        # the input (the ReLU takes care of the padding!)
        manual = input_numpy[0, 0,
                             relu(loc[0] - 1):loc[0] + 2,
                             relu(loc[1] - 1):loc[1] + 2,
                             relu(loc[2] - 1):loc[2] + 2,
                             relu(loc[3] - 1):loc[3] + 2].sum()

        # Print comparison
        print(f'At {tuple(loc)}:')
        print(f'\tconv4d:\t{conv4d}')
        print(f'\tmanual:\t{manual}')

    print()
