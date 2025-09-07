from typing import List
import torch
from torch import LongTensor, Tensor
import numpy as np


class BatchQueue:
    dims_shape = (3,)
    def __init__(self, dims: Tensor):
        """
        dims: [3] := (L,B,F), L>=1, B>=1, F>=1
        """
        assert dims.numel() == 3, "dims must be a tensor of length 3"

        self.L, self.B, self.F = map(int, dims.tolist())

        assert self.L >= 1 and self.B >= 1 and self.F >= 1, "All dimensions must be >= 1"

        self.counters = torch.zeros(self.B, dtype=torch.long)
        self.queue = torch.zeros((self.L, self.B, self.F), dtype=torch.float32)


    def dequeue(self, batch_indices: LongTensor) -> Tensor:
        """
        batch_indices: [N], 1<=N<=B
        Can assume there's at least one element at each batch index
        returns: [N,F]
        """
        ret_val = torch.zeros((len(batch_indices), self.F), dtype=torch.float32)

        for i, batch_index in enumerate(batch_indices):
            ret_val[i] = self.queue[0, batch_index]
            self.counters[batch_index] = self.counters[batch_index] - 1
            self.queue[0, batch_index] = torch.zeros(self.F, dtype=torch.float32)
            self.queue[:, batch_index] = torch.roll(self.queue[:, batch_index], shifts=-1, dims=0)

        return ret_val

    def enqueue(self, values: Tensor, batch_indices: LongTensor) -> None:
        """
        values: [T,N,F], 1<=T<=L, 1<=N<=B, F>=1
        batch_indices: [N]
        Can assume there's space left in each of index
        """

        T, B, F = values.shape
        for idx, batch_idx in enumerate(batch_indices):
            batch_next_idx = self.counters[batch_idx]
            self.queue[batch_next_idx : batch_next_idx + T, batch_idx] = values[:,idx]
            self.counters[batch_idx] += T


        
    def peek(self, location: List[str], batch_indices: LongTensor) -> Tensor:
        """
        location: [N], 1<=N<=B
        batch_indices: [N]
        Can return any value if the queue is empty
        returns: [N,F]
        """
        ret_val = torch.zeros((len(location), self.F), dtype=torch.float32)
        for i, (l, b) in enumerate(zip(location, batch_indices)):
            b = int(b)
            if l == 'head':
                ret_val[i] = self.queue[0,b]
            elif l == 'tail':
                batch_tail = self.counters[b]
                if batch_tail == 0:
                    ret_val[i] = self.queue[0, b]
                else:
                    ret_val[i] = self.queue[batch_tail - 1, b]

        return ret_val

               

def sanity_check():
    ### Keep in mind that these checks are only basic helpers and do not cover all cases. ###
    
    with open("./mel_spectrograms.npy", "rb") as f:
        mels = torch.tensor(np.load(f))
        
    expected_results = [
        torch.cat([
            mels[0, 0, :].unsqueeze(0),
            mels[4, 1, :].unsqueeze(0),
            torch.zeros([1,240000]),
        ]),
        torch.cat([
            mels[0, 0, :].unsqueeze(0),
            mels[0, 3, :].unsqueeze(0),
        ]),
        torch.cat([
            mels[1, 0, :].unsqueeze(0),
            mels[1, 3, :].unsqueeze(0),
            mels[4, 2, :].unsqueeze(0),
        ]),
        mels[4, 0, :].unsqueeze(0),
    ]
    results = []
        
    q = BatchQueue(torch.Tensor([10, 5, 240000]))

    q.enqueue(mels, batch_indices=torch.tensor([0, 1, 2, 4]))
    results.append(q.peek(['head', 'tail', 'head'], batch_indices=torch.tensor([0, 1, 3])))
    results.append(q.dequeue(batch_indices=torch.tensor([0, 4])))
    results.append(q.peek(['head', 'head', 'tail'], batch_indices=torch.tensor([0, 4, 2])))
    results.append(q.peek(['tail'], batch_indices=torch.tensor([0])))

    for i, (res, expected_res) in enumerate(zip(results, expected_results)):
        if not torch.equal(res, expected_res):
            raise AssertionError(f"Wrong results at index {i}")


if __name__ == "__main__":
    sanity_check()
