from torch import Tensor
from typing import Callable, Union, Any
from dataclasses import fields

class DeviceShiftable:
    def apply_to_tensors(self, func: Callable[[Tensor], Tensor]):
        '''
            Applies a function to all tensors in the dataclass.
        '''
        def apply(attr: Union[list, dict, Tensor, Any]):
            if isinstance(attr, Tensor):
                return func(attr)
            elif isinstance(attr, list):
                return [apply(x) for x in attr]
            elif isinstance(attr, dict):
                return {k : apply(v) for k, v in attr.items()}
            else:
                return attr

        for field in fields(self):
            attr = getattr(self, field.name)
            setattr(self, field.name, apply(attr))

        return self

    def to(self, device):
        '''
            Shifts all tensors to the specified device
        '''
        return self.apply_to_tensors(lambda x: x.to(device))

    def cpu(self):
        '''
            Moves all tensors to the CPU.
        '''
        return self.to('cpu')

    def cuda(self):
        '''
            Moves all tensors to the GPU.
        '''
        return self.to('cuda')

    def detach(self):
        '''
            Detaches all tensors.
        '''
        return self.apply_to_tensors(lambda x: x.detach())