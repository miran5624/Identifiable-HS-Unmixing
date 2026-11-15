import torch
from torch import nn, optim
import spectral

class AE(nn.Module):
    def __init__(self, b, p):
        super().__init__()
        self.e = nn.Sequential(
            nn.Linear(b, b // 2),
            nn.ReLU(),
            nn.Linear(b // 2, p),
            nn.Softmax(dim=1)
        )
        self.d = nn.Linear(p, b, bias=False)

    def forward(self, x):
        a = self.e(x)
        return self.d(a), a

def run():
    i = spectral.envi.open('data/Samson Dataset/samson_1.img.hdr', 'data/Samson Dataset/samson_1.img')
    x = torch.tensor(i.load().reshape(-1, 156), dtype=torch.float32)
    
    b, p = 156, 3
    m = AE(b, p)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    f = nn.MSELoss()
    
    for epoch in range(500):
        opt.zero_grad()
        x_h, a = m(x)
        l = f(x_h, x)
        l.backward()
        opt.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {l.item():.6f}")

    with torch.no_grad():
        x_h, a = m(x)

    torch.save(m.state_dict(), 'm.pt')
    torch.save({'x': x, 'x_h': x_h, 'a': a, 'e': m.d.weight.detach(), 'final_loss': f(x_h, x).item()}, 'r.pt')
    
    print("Training completed!\nModel saved as m.pt\nResults saved as r.pt")

if __name__ == '__main__':
    run()