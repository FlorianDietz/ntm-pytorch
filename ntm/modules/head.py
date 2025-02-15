import torch
import torch.nn.functional as F
from torch import nn


class NTMHead(nn.Module):

    def __init__(self, mode, controller_size, key_size, comgra, head_name):
        super().__init__()
        self.mode = mode
        self.key_size = key_size

        # all the fc layers to produce scalars for memory addressing
        self.key_fc = nn.Linear(controller_size, key_size)
        self.key_strength_fc = nn.Linear(controller_size, 1)

        # these five fc layers cannot be in controller class
        # since each head has its own parameters and scalars
        self.interpolation_gate_fc = nn.Linear(controller_size, 1)
        self.shift_weighting_fc = nn.Linear(controller_size, 3)
        self.sharpen_factor_fc = nn.Linear(controller_size, 1)
        # --(optional : for separation of add and erase mechanism)
        # self.erase_weight_fc = nn.Linear(controller_size, key_size)

        # fc layer to produce write data. data vector length=key_size
        self.write_data_fc = nn.Linear(controller_size, key_size)
        self.reset()

        self.comgra = comgra
        self.head_name = head_name

    def forward(self, controller_state, prev_weights, memory, data=None):
        """Accept previous state (weights and memory) and controller state,
        produce attention weights for current read or write operation.
        Weights are produced by content-based and location-based addressing.

        Refer *Figure 2* in the paper to see how weights are produced.

        The head returns current weights useful for next time step, while
        it reads from or writes to ``memory`` based on its mode, using the
        ``data`` vector. ``data`` is filled and returned for read mode,
        returned as is for write mode.

        Refer *Section 3.1* for read mode and *Section 3.2* for write mode.

        Parameters
        ----------
        controller_state : torch.Tensor
            Long-term state of the controller.
            ``(batch_size, controller_size)``

        prev_weights : torch.Tensor
            Attention weights from previous time step.
            ``(batch_size, memory_units)``

        memory : ntm_modules.NTMMemory
            Memory Instance. Read write operations will be performed in place.

        data : torch.Tensor
            Depending upon the mode, this data vector will be used by memory.
            ``(batch_size, memory_unit_size)``

        Returns
        -------
        current_weights, data : torch.Tensor, torch.Tensor
            Current weights and data (filled in read operation else as it is).
            ``(batch_size, memory_units), (batch_size, memory_unit_size)``
        """

        # all these are marked as "controller outputs" in Figure 2
        key = self.key_fc(controller_state)
        b_pre = self.key_strength_fc(controller_state)
        g_pre = self.interpolation_gate_fc(controller_state)
        s_pre = self.shift_weighting_fc(controller_state)
        if self.comgra is not None and self.comgra.RECORD_YES_NO:
            self.comgra.register_tensor(
                f"{self.head_name}_{self.comgra.ITERATION}_b", b_pre.clone().detach(), recording_type='neurons', is_target=True,
                node_name=f"head_b", role_within_node=f"{self.head_name}_{self.comgra.ITERATION}_b"
            )
            self.comgra.register_tensor(
                f"{self.head_name}_{self.comgra.ITERATION}_g", g_pre.clone().detach(), recording_type='neurons', is_target=True,
                node_name=f"head_g", role_within_node=f"{self.head_name}_{self.comgra.ITERATION}_g"
            )
            self.comgra.register_tensor(
                f"{self.head_name}_{self.comgra.ITERATION}_s", s_pre.clone().detach(), recording_type='neurons', is_target=True,
                node_name=f"head_s", role_within_node=f"{self.head_name}_{self.comgra.ITERATION}_s"
            )
        if self.comgra is not None and self.comgra.REGULARIZATION:
            for tensor_to_regularize in [g_pre, s_pre]:
                target = tensor_to_regularize.clamp(min=-1 * self.comgra.REGULARIZATION, max=self.comgra.REGULARIZATION).detach()
                loss_granular = torch.nn.functional.mse_loss(tensor_to_regularize, target, reduction='none')
                regularization_loss = loss_granular.mean()
                self.comgra.REGULARIZATION_LOSSES.append(regularization_loss)
        b = F.softplus(b_pre)
        g = F.sigmoid(g_pre)
        s = F.softmax(s_pre)
        # here the sharpening factor is less than 1 whereas as required in the
        # paper it should be greater than 1. hence adding 1.
        y = 1 + F.softplus(self.sharpen_factor_fc(controller_state))
        # e = F.sigmoid(self.erase_weight_fc(controller_state))  # erase vector
        a = self.write_data_fc(controller_state)  # add vector

        content_weights = memory.content_addressing(key, b)
        # location-based addressing - interpolate, shift, sharpen
        interpolated_weights = g * content_weights + (1 - g) * prev_weights
        shifted_weights = self._circular_conv1d(interpolated_weights, s)
        # the softmax introduces the exp of the argument which isn't there in
        # the paper. there it's just a simple normalization of the arguments.
        current_weights = shifted_weights ** y
        # current_weights = F.softmax(shifted_weights ** y)
        current_weights = torch.div(current_weights, torch.sum(
            current_weights, dim=1).view(-1, 1) + 1e-16)

        if self.mode == 'r':
            data = memory.read(current_weights)
        elif self.mode == 'w':
            # memory.write(current_weights, a, e)
            memory.write(current_weights, a)
        else:
            raise ValueError("mode must be read ('r') or write('w')")
        return current_weights, data

    @staticmethod
    def _circular_conv1d(in_tensor, weights):
        # pad left with elements from right, and vice-versa
        batch_size = weights.size(0)
        pad = int((weights.size(1) - 1) / 2)
        in_tensor = torch.cat(
            [in_tensor[:, -pad:], in_tensor, in_tensor[:, :pad]], dim=1)
        out_tensor = F.conv1d(in_tensor.view(batch_size, 1, -1),
                              weights.view(batch_size, 1, -1))
        out_tensor = out_tensor.view(batch_size, -1)
        return out_tensor

    def reset(self):
        nn.init.xavier_uniform_(self.key_strength_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.interpolation_gate_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.shift_weighting_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.sharpen_factor_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.write_data_fc.weight, gain=1.4)
        # nn.init.xavier_uniform_(self.erase_weight_fc.weight, gain=1.4)

        # nn.init.kaiming_uniform_(self.key_strength_fc.weight)
        # nn.init.kaiming_uniform_(self.interpolation_gate_fc.weight)
        # nn.init.kaiming_uniform_(self.shift_weighting_fc.weight)
        # nn.init.kaiming_uniform_(self.sharpen_factor_fc.weight)
        # nn.init.kaiming_uniform_(self.write_data_fc.weight)
        # nn.init.kaiming_uniform_(self.erase_weight_fc.weight)

        nn.init.normal_(self.key_fc.bias, std=0.01)
        nn.init.normal_(self.key_strength_fc.bias, std=0.01)
        nn.init.normal_(self.interpolation_gate_fc.bias, std=0.01)
        nn.init.normal_(self.shift_weighting_fc.bias, std=0.01)
        nn.init.normal_(self.sharpen_factor_fc.bias, std=0.01)
        nn.init.normal_(self.write_data_fc.bias, std=0.01)
        # nn.init.normal_(self.erase_weight_fc.bias, std=0.01)
