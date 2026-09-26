import torch
import flashinfer
from flashinfer.sampling import top_k_top_p_sampling_from_logits

logits = torch.randn(1, 100).cuda().half()
k = torch.tensor([10]).cuda()
p = torch.tensor([0.9]).cuda()
out = top_k_top_p_sampling_from_logits(logits, k, p)
print(out)
