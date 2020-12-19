import torch

new = torch.load('../new.pt')
old = torch.load('../old.pt')

for k in new:
    if k in old:
        new[k] = old[k].view(new[k].shape)
        print(f'Copied {k}')

torch.save(new, '../new.pt')
