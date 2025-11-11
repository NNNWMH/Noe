import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 物理常数 ----------
L = 0.1
c = 4.7e-8

# ---------- 原始数据 ----------
f  = np.arange(1600, 3000, 100)
U  = np.array([88.9, 106.4, 129.9, 163.2, 213.2, 294.2, 427.6, 564.9,
               495.1, 354.8, 263.1, 206.9, 170.2, 144.7, 126.0])
U2 = np.array([401.9, 461.8, 531.3, 610.1, 694.8, 776.7, 840.3, 868.9,
               856.3, 811.2, 749.6, 684.2, 622.6, 567.5, 519.3])

f1 = np.arange(2220, 2380, 10)
U1 = np.array([460.8, 477.3, 493.5, 509.2, 523.9, 537.2, 548.9, 558.4, 565.4,
               569.6, 570.8, 569.1, 564.6, 557.5, 548.1, 536.7, 523.8])
U3 = np.array([849.3, 853.3, 856.8, 859.9, 862.6, 864.8, 866.7, 868.1, 869.1,
               869.6, 869.8, 869.5, 868.8, 867.7, 866.2, 864.3, 862.0])

# ---------- 电流计算 ----------
def I_mA(U_mV, f_Hz, R_Ohm):
    w = 2 * np.pi * f_Hz
    return (U_mV / 1000) / np.sqrt(R_Ohm**2 + (w*L - 1/(w*c))**2) * 1000   # mV→mA

I_100  = I_mA(U,  f,  100)
I_500  = I_mA(U2, f,  500)
I1_100 = I_mA(U1, f1, 100)
I1_500 = I_mA(U3, f1, 100)

# 1 Hz 插值
f_fine  = np.arange(f.min(), f.max() + 1, 1)
f1_fine = np.arange(f1.min(), f1.max() + 1, 1)
I_100_f = np.interp(f_fine, f, I_100)
I_500_f = np.interp(f_fine, f, I_500)
I1_100_f = np.interp(f1_fine, f1, I1_100)
I1_500_f = np.interp(f1_fine, f1, I1_500)

# ---------- 2320 Hz 物理带宽思路 ----------
f_ref = 2320
R_list = [100, 500]          # 两条曲线
I_given = [570.8, 869.8]     # 给定电流（mA）

for R, I_raw in zip(R_list, I_given):
    I_half = I_raw / np.sqrt(2)                 # 半功率电流
    I_curve = I_100_f if R == 100 else I_500_f

    # 找 2320 左右各 1 点
    idx_c = np.argmin(np.abs(f_fine - f_ref))
    idx_l = max(idx_c - 1, 0)
    idx_r = min(idx_c + 1, len(f_fine) - 1)

    # 左右电流 & 频率
    f_left,  I_left  = f_fine[idx_l], I_curve[idx_l]
    f_right, I_right = f_fine[idx_r], I_curve[idx_r]

    # 带宽
    delta_f = f_right - f_left

    # 输出
    print(f'{R} Ω  2320 Hz 给定 I = {I_raw} mA → I/√2 = {I_half:.3f} mA')
    print(f'  左右最接近点：')
    print(f'    f = {f_left:.0f} Hz,  I = {I_left:.3f} mA')
    print(f'    f = {f_right:.0f} Hz,  I = {I_right:.3f} mA')
    print(f'  → 带宽 Δf = {delta_f:.0f} Hz\n')

# ---------- 图 1：独立纵轴（带宽真实比例） ----------
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

l1 = ax1.plot(f_fine, I_100_f, color='b', label='R = 100 Ω')
l2 = ax2.plot(f_fine, I_500_f, color='g', label='R = 500 Ω')

# 2320 Hz 基准线
ax1.axhline(I_given[0]/np.sqrt(2), color='b', ls=':', alpha=0.7)
ax2.axhline(I_given[1]/np.sqrt(2), color='g', ls=':', alpha=0.7)

# 最接近点
for ff, ii in zip(f_fine[[np.argmin(np.abs(f_fine-2320))-1, np.argmin(np.abs(f_fine-2320))+1]],
                  [I_100_f[np.argmin(np.abs(f_fine-2320))-1], I_100_f[np.argmin(np.abs(f_fine-2320))+1]]):
    ax1.scatter(ff, ii, color='b', marker='x', s=40, zorder=5)
for ff, ii in zip(f_fine[[np.argmin(np.abs(f_fine-2320))-1, np.argmin(np.abs(f_fine-2320))+1]],
                  [I_500_f[np.argmin(np.abs(f_fine-2320))-1], I_500_f[np.argmin(np.abs(f_fine-2320))+1]]):
    ax2.scatter(ff, ii, color='g', marker='x', s=40, zorder=5)

ax1.set_xlabel('频率 f (Hz)')
ax1.set_ylabel('100 Ω 电流 I (mA)', color='b')
ax2.set_ylabel('500 Ω 电流 I (mA)', color='g')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')
ax1.set_title('2320 Hz 给定电流 /√2 最接近两点（独立纵轴）')
ax1.grid(True)
labs = [l.get_label() for l in l1+l2]
ax1.legend(labs, loc='upper right')
plt.tight_layout()

# ---------- 图 2：谐振附近 ----------
plt.figure(2, figsize=(6, 4))
plt.plot(f1_fine, I1_100_f, label='R = 100 Ω', color='b')
plt.plot(f1_fine, I1_500_f, label='R = 500 Ω', color='g')
plt.xlabel('频率 f (Hz)')
plt.ylabel('电流 I (mA)')
plt.title('表二：I - f 谐振附近（1 Hz 插值）')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()