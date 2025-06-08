import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ─── 1. Util ──────────────────────────────────────────────
class BinFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.sign()
    @staticmethod
    def backward(ctx, g): return g
sign_ste = BinFn.apply

class FakeQuant8(nn.Module):            # Conv1용
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, w):               # 학습·추론용 float 출력
        with torch.no_grad():
            self.scale.copy_(w.abs().max() / 127 + 1e-8)
        q = torch.clamp((w / self.scale).round(), -128, 127)
        return q * self.scale

    def quantize_to_int(self, w):       # HLS용 int8 추출
        with torch.no_grad():
            self.scale.copy_(w.abs().max() / 127 + 1e-8)
        return torch.clamp((w / self.scale).round(), -128, 127).to(torch.int8)

class HardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x ≥ 0 은 +1, x < 0 은 −1
        return torch.where(x >= 0,
                           torch.ones_like(x),
                           -torch.ones_like(x))
    @staticmethod
    def backward(ctx, g):
        return g

hsign = HardSign.apply

# ─── 2. TeacherNet (FP32) ─────────────────────────────────
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool  = nn.AvgPool2d(2)
        self.fc    = nn.Linear(32*11*11, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ─── 3. Binary Convolution (일반 BNN) ─────────────────────
class BConv2d(nn.Conv2d):
    def forward(self, x):
        w_bin = hsign(self.weight)
        return F.conv2d(x, w_bin, None, self.stride, self.padding)

# ─── 4. StudentQAT ────────────────────────────────────────
class StudentQAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, bias=False)
        self.q1    = FakeQuant8()              # Conv1 weight 양자화

        self.conv2 = BConv2d(16, 16, 3, bias=False)
        self.conv3 = BConv2d(16, 32, 3, bias=False)
        self.pool  = nn.AvgPool2d(2)
        self.fc    = nn.Linear(32*11*11, 10, bias=True)

    def forward(self, x):
        # ─ Conv1: 학습 중 양자화된 int8 weight 사용
        w_int8 = self.q1.quantize_to_int(self.conv1.weight)         # torch.int8
        w_q    = w_int8.to(torch.float32)
        x = F.conv2d(x, w_q, None, self.conv1.stride, self.conv1.padding)
        x = hsign(x)

        # ─ Conv2,3: BNN
        x = hsign(self.conv2(x))
        x = hsign(self.conv3(x))

        # ─ Pool → FC
        x = self.pool(x).flatten(1)
        w_fc = hsign(self.fc.weight)
        return F.linear(x, w_fc, self.fc.bias)

# ─── 5. KD Loss ───────────────────────────────────────────
def kd_loss(s, t, y, T=1.0, alpha=0.3):
    ce = F.cross_entropy(s, y)
    kd = F.kl_div(F.log_softmax(s/T, 1), F.softmax(t/T, 1),
                  reduction="batchmean") * T * T
    return alpha*kd + (1-alpha)*ce

# ─── 6. Train / Eval 루프 ────────────────────────────────
def train_ep(stu, tea, ldr, opt, ep, T, alpha):
    stu.train(); tea.eval(); tot=hit=loss_sum=0
    for x, y in ldr:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.no_grad(): t = tea(x)
        s = stu(x)
        loss = kd_loss(s, t, y, T, alpha)
        loss.backward(); opt.step()
        tot += y.size(0); hit += (s.argmax(1)==y).sum().item(); loss_sum += loss.item()
    print(f"Epoch {ep:02d} | loss {loss_sum/len(ldr):.4f} | acc {100*hit/tot:.2f}%")
    return loss_sum/len(ldr), 100*hit/tot

@torch.no_grad()
def eval_acc(net, ldr, tag):
    net.eval(); tot=hit=0
    for x, y in ldr:
        x, y = x.to(device), y.to(device)
        hit += (net(x).argmax(1)==y).sum().item(); tot += y.size(0)
    acc = 100*hit/tot
    print(f"{tag} acc {acc:.2f}%"); return acc

# ─── 7. 데이터 ────────────────────────────────────────────
def binarize_img(t): return (t>0.5).float()*2-1
tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(binarize_img)])
def get_ld(bs=128):
    tr = datasets.MNIST(".", True , download=True, transform=tfm)
    te = datasets.MNIST(".", False, download=True, transform=tfm)
    return (DataLoader(tr, bs, True , num_workers=2, pin_memory=True),
            DataLoader(te, bs, False, num_workers=2, pin_memory=True))

# ─── 8. HLS 검증 ─────────────────────────────────────────
def pack_bits(img):
    bits = (img > 0.5).to(torch.uint8)
    out  = torch.empty((img.size(0), 98), dtype=torch.uint8)
    for i in range(img.size(0)):
        out[i] = torch.from_numpy(np.packbits(bits[i].view(-1).cpu().numpy()))
    return out

def unpack_to_pm1(packed):
    imgs = []
    for row in packed:
        bits = np.unpackbits(row.cpu().numpy())[:784].astype(np.float32)
        imgs.append(torch.tensor(bits.reshape(1,28,28))*2 - 1)
    return torch.stack(imgs).to(device)

@torch.no_grad()
def verify_model(model, loader):
    model.eval()
    hit=tot=0
    for x, y in loader:
        pk   = pack_bits(x)
        x_pm = unpack_to_pm1(pk)
        y    = y.to(device)
        logits = model(x_pm)
        hit += (logits.argmax(1)==y).sum().item()
        tot += y.size(0)
    acc = 100*hit/tot
    print(f"📊 HLS style acc = {acc:.2f}%")
    return acc

# ─── 9. 가중치 저장 ───────────────────────────────────────
@torch.no_grad()
def save_weights_correctly(student, save_path):
    student.eval()
    print("\n🔧 가중치 저장 중...")

    # Conv1
    conv1_w = student.conv1.weight.data
    conv1_q = student.q1.quantize_to_int(conv1_w).cpu().numpy().astype(np.int8)
    conv1_scale = student.q1.scale.item()
    print(f"Conv1 scale = {conv1_scale:.6f}, quantized range = [{conv1_q.min()}, {conv1_q.max()}]")

    # Conv2/3/FC 이진화
    def binarize(w, name):
        s = w.sign().cpu().numpy().astype(np.int8)
        b = ((s + 1)//2).astype(np.uint8)
        print(f"{name} binary unique = {np.unique(b)}")
        return b

    conv2_q = binarize(student.conv2.weight, "Conv2")
    conv3_q = binarize(student.conv3.weight, "Conv3")
    fc_w_q  = binarize(student.fc.weight,  "FC")

    fc_b    = student.fc.bias.cpu().numpy().astype(np.float32)
    print(f"FC bias range = [{fc_b.min():.3f}, {fc_b.max():.3f}]")

    weights = {
        "conv1": conv1_q,
        "conv2": conv2_q,
        "conv3": conv3_q,
        "fc_w":  fc_w_q,
        "fc_b":  fc_b,
    }
    np.save(save_path, weights)
    print(f"✅ 저장 완료: {save_path}")
    return weights

# ─── 10. 메인 ────────────────────────────────────────────
if __name__ == "__main__":
    tr_loader, te_loader = get_ld()

    # 1) Teacher 학습
    teacher = TeacherNet().to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), 1e-3)
    for ep in range(10):
        train_ep(teacher, teacher, tr_loader, opt_t, ep, 1, 0)
    eval_acc(teacher, te_loader, "Teacher")

    # 2) Student 학습
    student = StudentQAT().to(device)
    opt_s = torch.optim.Adam(student.parameters(), 1e-3)
    best_acc=0
    for ep in range(70):
        train_ep(student, teacher, tr_loader, opt_s, ep, 1.0, 0.3)
        acc = eval_acc(student, te_loader, "Student")
        hls = verify_model(student, te_loader)
        best_acc = max(best_acc, acc)
    print(f"✅ Best Student acc = {best_acc:.2f}%")

    # 3) 가중치 저장
    save_path = "/home/jinseopalang/young/202506081012.npy"
    save_weights_correctly(student, save_path)

    # 4) 최종 검증
    final_acc = eval_acc(student, te_loader, "Final")
    final_hls = verify_model(student, te_loader)
    print(f"📈 최종: normal={final_acc:.2f}%, hls={final_hls:.2f}%")
