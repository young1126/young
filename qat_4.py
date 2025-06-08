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
    """일반적인 BNN: sign() 적용된 가중치로 convolution"""
    
    def forward(self, x):
        w_bin = sign_ste(self.weight)  # 가중치만 이진화
        return F.conv2d(x, w_bin, None, self.stride, self.padding)

# ─── 4. StudentQAT (일반 BNN 방식) ────────────────────────
class StudentQAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, bias=False)
        self.q1    = FakeQuant8()              # 8-bit quantization
        self.conv2 = BConv2d(16, 16, 3, bias=False)  # Binary weights
        self.conv3 = BConv2d(16, 32, 3, bias=False)  # Binary weights
        self.pool  = nn.AvgPool2d(2)
        self.fc    = nn.Linear(32*11*11, 10, bias=True)

    def forward(self, x):
        # Conv1: 8비트 양자화 + Sign 활성화
        x = self.q1(self.conv1(x))  # 8비트 양자화된 convolution
        x = sign_ste(x)             # Sign 활성화
        
        # Conv2: Binary weights + Sign 활성화
        x = sign_ste(self.conv2(x))
        
        # Conv3: Binary weights + Sign 활성화  
        x = sign_ste(self.conv3(x))
        
        # Pooling + FC
        x = self.pool(x).flatten(1)
        
        # FC: Binary weights
        w_fc = sign_ste(self.fc.weight)
        return F.linear(x, w_fc, self.fc.bias)

# ─── 5. KD Loss ───────────────────────────────────────────
def kd_loss(s, t, y, T=2.0, alpha=0.5):
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

# ─── 8. 검증 함수 ────────────────────────────────────────
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
    """HLS 방식 검증: 비트패킹 입력 사용"""
    model.eval()
    
    hit = tot = 0
    for x, y in loader:
        pk   = pack_bits(x)
        x_pm = unpack_to_pm1(pk)
        y    = y.to(device)

        logits = model(x_pm)
        pred = logits.argmax(1)
        hit += (pred == y).sum().item()
        tot += y.size(0)

    acc = 100*hit/tot
    print(f"📊 HLS style acc = {acc:.2f}%")
    return acc

# ─── 9. 올바른 가중치 저장 ─────────────────────────────────
@torch.no_grad()
def save_weights_correctly(student, save_path):
    """HLS와 완전히 매칭되는 가중치 저장"""
    student.eval()
    
    print("\n🔧 가중치 저장 중...")
    
    # 1. Conv1: 8비트 양자화 + 스케일
    conv1_w = student.conv1.weight.data
    conv1_quantized = student.q1.quantize_to_int(conv1_w).cpu().numpy()
    conv1_scale = student.q1.scale.cpu().item()
    
    print(f"Conv1 원본 범위: [{conv1_w.min():.3f}, {conv1_w.max():.3f}]")
    print(f"Conv1 양자화 범위: [{conv1_quantized.min()}, {conv1_quantized.max()}]")
    print(f"Conv1 스케일: {conv1_scale:.6f}")
    
    # 2. Conv2, Conv3, FC: 1비트 이진화
    def to_binary_correct(weight, name):
        """올바른 이진화: sign() → {-1,+1} → {0,1}"""
        with torch.no_grad():
            # Step 1: 가중치에 sign() 적용
            sign_w = weight.sign()
            unique_sign = torch.unique(sign_w)
            print(f"{name} sign 유니크 값: {unique_sign}")
            
            # Step 2: {-1,+1} → {0,1} 변환
            binary_w = ((sign_w + 1) / 2).cpu().numpy().astype(np.uint8)
            unique_bin = np.unique(binary_w)
            print(f"{name} 이진화 유니크 값: {unique_bin}")
            
            # Step 3: 분포 확인
            ratio_1 = np.mean(binary_w == 1)
            ratio_0 = np.mean(binary_w == 0)
            print(f"{name} 분포 - 0: {ratio_0:.3f}, 1: {ratio_1:.3f}")
            
            # 문제 체크
            if not np.array_equal(unique_bin, [0, 1]):
                print(f"⚠️  {name} 이진화 실패! 유니크 값: {unique_bin}")
            
            return binary_w
    
    conv2_bin = to_binary_correct(student.conv2.weight, "Conv2")
    conv3_bin = to_binary_correct(student.conv3.weight, "Conv3") 
    fc_w_bin = to_binary_correct(student.fc.weight, "FC")
    
    # 3. FC bias: float32
    fc_bias = student.fc.bias.cpu().numpy().astype(np.float32)
    print(f"FC bias 범위: [{fc_bias.min():.3f}, {fc_bias.max():.3f}]")
    print(f"FC bias 평균: {fc_bias.mean():.3f}, 표준편차: {fc_bias.std():.3f}")
    
    # 4. 가중치 통계 확인
    print("\n📊 가중치 통계:")
    print(f"Conv1 non-zero 비율: {np.mean(conv1_quantized != 0):.3f}")
    print(f"Conv2 1의 비율: {np.mean(conv2_bin == 1):.3f}")
    print(f"Conv3 1의 비율: {np.mean(conv3_bin == 1):.3f}")
    print(f"FC 1의 비율: {np.mean(fc_w_bin == 1):.3f}")
    
    # 5. 저장
    weights_dict = {
        "conv1": conv1_quantized,
        "conv1_scale": conv1_scale,
        "conv2": conv2_bin,
        "conv3": conv3_bin,
        "fc_w": fc_w_bin,
        "fc_b": fc_bias
    }
    
    np.save(save_path, weights_dict)
    print(f"✅ 가중치 저장 완료: {save_path}")
    
    # 6. 저장 검증
    print("\n🔍 저장된 가중치 검증:")
    saved = np.load(save_path, allow_pickle=True).item()
    for key, value in saved.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"    범위: [{value.min()}, {value.max()}]")
            if key in ['conv2', 'conv3', 'fc_w']:
                unique_vals = np.unique(value)
                print(f"    유니크 값: {unique_vals}")
                if not np.array_equal(unique_vals, [0, 1]):
                    print(f"    ❌ 예상과 다름! 0,1이어야 함")
                else:
                    print(f"    ✅ 올바른 이진 값")
        else:
            print(f"  {key}: {value}")
        print()
    
    return weights_dict

# ─── 10. 메인 ────────────────────────────────────────────
if __name__ == "__main__":
    tr_loader, te_loader = get_ld()

    # ① Teacher 학습
    print("🎓 Teacher 학습 시작")
    teacher = TeacherNet().to(device)
    opt_t = torch.optim.Adam(teacher.parameters(), 1e-3)
    
    for ep in range(10):
        loss, acc = train_ep(teacher, teacher, tr_loader, opt_t, ep, 1, 0)
        
        # 학습이 제대로 안되면 조기 종료
        if ep > 5 and acc < 50:
            print("⚠️  Teacher 학습이 제대로 안됩니다. Learning rate 조정 필요.")
            break
    
    teacher_acc = eval_acc(teacher, te_loader, "Teacher")
    
    # Teacher 성능 체크
    if teacher_acc < 85:
        print(f"⚠️  Teacher 성능이 낮습니다: {teacher_acc:.1f}%")
        print("더 학습하거나 모델 구조를 개선하세요.")
        # 그래도 계속 진행
    
    # ② Student 학습
    print("\n🎓 Student QAT 학습 시작")
    student = StudentQAT().to(device)
    opt_s = torch.optim.Adam(student.parameters(), 1e-3)
    
    best_acc = 0
    best_epoch = 0
    
    for ep in range(40):
        loss, acc = train_ep(student, teacher, tr_loader, opt_s, ep, 1.0, 0.3)
        val_acc = eval_acc(student, te_loader, "Student")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = ep
        
        # 학습 진행 체크
        if ep > 15 and val_acc < 20:
            print("❌ Student 학습이 실패하고 있습니다!")
            print("파라미터 조정이 필요합니다:")
            print("  - Learning rate 감소 (1e-4)")
            print("  - Temperature 증가 (5.0)")
            print("  - Alpha 조정 (0.8)")
            break
        
        # 검증
        if ep % 10 == 0 or ep == 39:
            print(f"\n--- Epoch {ep} 검증 ---")
            hls_acc = verify_model(student, te_loader)
            print(f"일반 정확도: {val_acc:.2f}%")
            print(f"HLS 정확도: {hls_acc:.2f}%")
            print(f"차이: {abs(val_acc - hls_acc):.2f}%")
            print("--- 검증 완료 ---\n")
    
    print(f"✅ Best Student acc: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # ③ 최종 성능 체크
    if best_acc < 60:
        print("❌ Student 성능이 너무 낮습니다!")
        print("가중치 저장을 진행하지만, 학습을 다시 해보세요.")
    
    # ④ 가중치 저장
    print("\n💾 가중치 저장")
    save_path = "/home/jinseopalang/young/hls_friendly_weights_fixed.npy"
    weights = save_weights_correctly(student, save_path)
    
    # ⑤ 최종 검증
    print("\n🎯 최종 검증:")
    final_normal_acc = eval_acc(student, te_loader, "Final")
    final_hls_acc = verify_model(student, te_loader)
    
    print(f"\n📈 최종 결과:")
    print(f"일반 정확도: {final_normal_acc:.2f}%")
    print(f"HLS 정확도: {final_hls_acc:.2f}%")
    print(f"차이: {abs(final_normal_acc - final_hls_acc):.2f}%")
    
    if final_hls_acc > 60:
        print(f"✅ 학습 성공!")
        print(f"💾 가중치 파일: {save_path}")
    else:
        print(f"⚠️  HLS 정확도가 낮습니다: {final_hls_acc:.2f}%")
        print("가중치는 저장되었지만 재학습을 권장합니다.")