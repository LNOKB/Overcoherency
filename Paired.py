#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os, datetime, csv
import numpy as np

from psychopy import gui, visual, core, data, event, logging, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)

from psychopy.tools.monitorunittools import deg2pix
try:
    import cv2
    _USE_CV2 = True
except Exception:
    _USE_CV2 = False
    from scipy.ndimage import gaussian_filter  # フォールバック（今回σ=0なので未使用）

# ===== 実験固定パラメータ =====
N_DOTS           = 120
DOT_LIFE         = 3
GAUSS_SIZE       = 0.005          # 'height'基準の直径（ドットの描画サイズ）
PRE_STIM_FIX_SEC = 1.0
ITI_SEC          = 0.5
RESP_TIMEOUT     = None           # 例: 3.0, Noneで無制限
field_diam       = 0.15
field_rad        = field_diam / 2.0
ISI_SEC          = 0.3            # First-Second 間 300ms
STIM_FRAMES      = 18             # 60Hzなら~300ms（適宜調整可）

# ROIサイズ（直径）。元が0.3だったので、1/3の 0.1 に設定
ROI_SIZE         = 0.10

# ===== 参加者情報 =====
dlg = gui.Dlg(title="Experiment Info")
dlg.addField("Participant ID:", "")
dlg.addField("Block Name:", "")
ok = dlg.show()
if not ok:
    core.quit()
participant_id = (dlg.data[0] or "unknown").strip()
block_name     = (dlg.data[1] or "block").strip()
  
block_num = int(block_name)
SPEED = 0.01 if (block_num % 2 == 1) else 0.0033

def sanitize(s):
    s = s.strip().replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))

pid_safe   = sanitize(participant_id)
block_safe = sanitize(block_name)
  
# ===== デザイン（42試行）=====
# vf_cond は {0,10} のみ。
# 0（中心）：coh ∈ {0.2, 0.5, 0.8}
# 10（周辺）：coh ∈ {0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0}
# 方向は {90, 270}。First/Second のどちらかが 0、もう一方が 10（必ず片方ずつ）
CENTER_COH  = [0.2, 0.5, 0.8]
PERIPH_COH  = [0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]
DIRS        = [90, 270]

trials = []
for d in DIRS:
    for c0 in CENTER_COH:
        for c10 in PERIPH_COH:
            first_is_center = np.random.rand() < 0.5
            if first_is_center:
                coh_first,  vf_first  = c0, 0
                coh_second, vf_second = c10, 10
            else:
                coh_first,  vf_first  = c10, 10
                coh_second, vf_second = c0, 0
            trials.append({
                'dir_deg': d,
                'coh_center': c0,
                'coh_periph': c10,
                'coh_first': coh_first,
                'vf_first': vf_first,
                'coh_second': coh_second,
                'vf_second': vf_second,
                'first_is_center': int(first_is_center)  # 1=Firstが中心, 0=Secondが中心
            })

np.random.shuffle(trials)
N_TRIALS = len(trials)  # = 42

# ===== 画面と刺激 =====
win = visual.Window(
    fullscr=True, units='height', color=-1, allowGUI=False,
    allowStencil=True, winType='pyglet', monitor='2b2_monitor'
)
fix = visual.TextStim(win, text='+', pos=(0,0), color=1.0, height=0.08)

# 応答プロンプト（Direction → Coherence の順に提示）
direction_label = visual.TextStim(
    win,
    text='Direction?\nUp  /  Down',
    pos=(0,0), color=1.0, height=0.08
)
coh_label = visual.TextStim(
    win,
    text='More Coherent?\n1  /  2',
    pos=(0,0), color=1.0, height=0.08
)

imgStim = visual.ImageStim(win, units='pix', size=win.size, interpolate=True)

# ===== 保存準備 =====
os.makedirs("data", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fname = f"data/RDK_{pid_safe}_{block_safe}_{timestamp}.csv"
fname_eye = f"data/RDK_eye_{pid_safe}_{block_safe}_{timestamp}.csv"
thisExp = data.ExperimentHandler(name='coh_cmp', version='',
    extraInfo=None, runtimeInfo=None, originPath='',
    savePickle=True, saveWideText=True, dataFileName=fname)
logFile = logging.LogFile(fname+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)
frameTolerance = 0.001

# 出力カラム（First/Secondの outframe を分離して保存 + Direction応答を追加）
fieldnames = [
    'participant_id','block_name','trial_index',
    'dir_deg',
    'coh_center','coh_periph',
    'coh_first','vf_first','coh_second','vf_second',
    'first_is_center',                         
    # --- Direction応答 ---
    'dir_resp_key','dir_resp', 'dir_resp_deg', 'dir_rt_sec',
    # --- Coherence(間隔)応答 ---
    'interval_choice_key','interval_choice','interval_rt_sec',
    # --- 刺激・視線 ---
    'n_dots','speed','dot_life','field_diam_used',
    'outframe_first','outframe_second'
]
rows_out = []

# ===== 入力・視線 =====
ioConfig = {
    'eyetracker.hw.tobii.EyeTracker': {
        'name': 'tracker', 'model_name': '', 'serial_number': '',
        'runtime_settings': {'sampling_rate': 60.0}
    },
    'Keyboard': dict(use_keymap='psychopy')
}
ioServer = iohub.launchHubServer(window=win, experiment_code='onlyeyetracking',
                                 datastore_name=fname_eye, **ioConfig)
eyetracker = ioServer.getDevice('tracker')
startRecording = hardware.eyetracker.EyetrackerControl(tracker=eyetracker, actionType='Start Only')

# ROIを1/3サイズ（直径0.10）に変更
roi = visual.ROI(
    win, name='roi', device=eyetracker, debug=False,
    shape='circle', pos=[0,0], size=(ROI_SIZE, ROI_SIZE), anchor='center', ori=0.0
)

# Create some handy timers
globalClock = core.Clock()
routineTimer = core.Clock()
# define target for calibration
calibrationTarget = visual.TargetStim(win, 
    name='calibrationTarget',
    radius=0.05, fillColor='', borderColor='black', lineWidth=2.0,
    innerRadius=0.017, innerFillColor='green', innerBorderColor='black', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for calibration
calibration = hardware.eyetracker.EyetrackerCalibration(win, 
    eyetracker, calibrationTarget,
    units=None, colorSpace='rgb',
    progressMode='time', targetDur=1.5, expandScale=1.5,
    targetLayout='THREE_POINTS', randomisePos=True, textColor='white',
    movementAnimation=True, targetDelay=1.0
)
# run calibration
calibration.run()
routineTimer.reset()

# ===== ローパス（今回は 0/10 ともに σ=0 に設定；位置の違いのみ反映）=====
def sigma_deg_for_vf(vf):
    return 0.0

# ===== ユーティリティ =====
def random_points_in_circle(n, radius):
    rho   = radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    return np.column_stack([rho*np.cos(theta), rho*np.sin(theta)])

def present_rdk_once(win, roi, coh, vf, dir_deg, field_rad):
    """
    与えられた (coh, vf, dir_deg) で STIM_FRAMES 枚表示。
    戻り値: (abort, outframe)
    """
    # 周辺(10)は左右どちらかにオフセット、中心(0)は中央
    if vf == 10:
        offset = np.random.choice([-0.35, 0.35])
    else:
        offset = 0.0
    dodge = np.array([offset, 0.0], dtype=float)

    # 初期化（この呼び出しは独立 RDK）
    pos = random_points_in_circle(N_DOTS, field_rad)
    n_signal = int(round(coh * N_DOTS))
    is_signal = np.zeros(N_DOTS, dtype=bool); is_signal[:n_signal] = True
    np.random.shuffle(is_signal)
    lifetimes = np.random.randint(1, DOT_LIFE + 1, size=N_DOTS)
    step_vec = SPEED * np.array([np.cos(np.deg2rad(dir_deg)), np.sin(np.deg2rad(dir_deg))])

    # ROI 初期化
    roi.setPos((0, 0))
    roi.reset()
    roi.status = NOT_STARTED
    roi.wasLookedIn = False
    roi.timesOn, roi.timesOff = [], []

    outframe = 0
    frameN = -1

    for _ in range(STIM_FRAMES):
        if event.getKeys(['escape']): return True, outframe
        frameN += 1

        # ROI bookkeeping（中央 ROIから外れていたフレーム数をカウント）
        if roi.isLookedIn:
            pass
        else:
            outframe += 1
        if roi.status == NOT_STARTED and frameN >= 0:
            roi.status = STARTED
            roi.clock.reset()
        if roi.status == STARTED:
            if roi.isLookedIn:
                if not roi.wasLookedIn:
                    roi.timesOn.append(roi.clock.getTime())
                    roi.timesOff.append(roi.clock.getTime())
                else:
                    roi.timesOff[-1] = roi.clock.getTime()
                roi.wasLookedIn = True
            else:
                if roi.wasLookedIn:
                    roi.timesOff[-1] = roi.clock.getTime()
                roi.wasLookedIn = False

        # ====== RDK 更新 ======
        lifetimes -= 1
        dead = lifetimes <= 0
        if np.any(dead):
            pos[dead] = random_points_in_circle(dead.sum(), field_rad)
            lifetimes[dead] = DOT_LIFE

        # シグナル移動
        pos[is_signal] += step_vec
        # ノイズ移動（ランダム方向）
        idx_noise = ~is_signal
        if np.any(idx_noise):
            phi = 2*np.pi*np.random.rand(idx_noise.sum())
            pos[idx_noise] += SPEED * np.column_stack([np.cos(phi), np.sin(phi)])

        # 枠外ヒット→円周に戻す
        r = np.hypot(pos[:,0], pos[:,1])
        outside = r > field_rad
        if np.any(outside):
            theta = np.arctan2(pos[outside,1], pos[outside,0]) + np.pi
            rr = field_rad * 0.999
            pos[outside] = np.column_stack([rr*np.cos(theta), rr*np.sin(theta)])
            lifetimes[outside] = DOT_LIFE

        # ====== ドット層 →（今回は σ=0 のため無加工）→ 表示 ======
        H, W = win.size[1], win.size[0]
        canvas = np.zeros((H, W), dtype=np.uint8)
        xy_h = pos + dodge
        xs = np.clip((W/2 + xy_h[:,0]*H).astype(np.int32), 0, W-1)
        ys = np.clip((H/2 - xy_h[:,1]*H).astype(np.int32), 0, H-1)
        dot_rad_px = max(1, int(round(GAUSS_SIZE * H / 2.0)))

        if _USE_CV2:
            for x, y in zip(xs, ys):
                cv2.circle(canvas, (int(x), int(y)), dot_rad_px, 255, -1, lineType=cv2.LINE_AA)
        else:
            yy, xx = np.ogrid[:H, :W]
            for x, y in zip(xs, ys):
                mask = (xx-x)**2 + (yy-y)**2 <= dot_rad_px**2
                canvas[mask] = 255

        # 円形アパーチャ
        center_x = int(round(W/2 + dodge[0]*H))
        center_y = int(round(H/2 - dodge[1]*H))
        field_rad_px = int(round(field_rad * H))
        if _USE_CV2:
            mask = np.zeros_like(canvas)
            cv2.circle(mask, (center_x, center_y), field_rad_px, 255, -1, lineType=cv2.LINE_AA)
            canvas = cv2.bitwise_and(canvas, mask)
        else:
            yy, xx = np.ogrid[:H, :W]
            circ = (xx-center_x)**2 + (yy-center_y)**2 <= field_rad_px**2
            canvas = np.where(circ, canvas, 0)

        blurred = canvas  # σ=0
        imgStim.image = np.dstack([blurred]*3).astype(np.float32) / 255.0
        imgStim.draw()
        fix.draw()
        win.flip()

    return False, outframe

# ===== 実験本体 =====
abort = False
for t_idx, tr in enumerate(trials, start=1):
    dir_deg         = tr['dir_deg']
    coh_center      = tr['coh_center']
    coh_periph      = tr['coh_periph']
    coh_first       = tr['coh_first']
    vf_first        = tr['vf_first']
    coh_second      = tr['coh_second']
    vf_second       = tr['vf_second']
    first_is_center = tr['first_is_center']

    # ---- 事前固視 ----
    pre_clock = core.Clock()
    while pre_clock.getTime() < PRE_STIM_FIX_SEC:
        if event.getKeys(['escape']):
            abort = True
            break
        fix.draw(); win.flip()
    if abort: break

    # ---- First 提示 ----
    a1, out1 = present_rdk_once(win, roi, coh_first, vf_first, dir_deg, field_rad)
    if a1: abort = True
    if abort: break

    # ---- ISI（300ms 固視）----
    isi_clock = core.Clock()
    while isi_clock.getTime() < ISI_SEC:
        if event.getKeys(['escape']):
            abort = True
            break
        fix.draw(); win.flip()
    if abort: break

    # ---- Second 提示 ----
    a2, out2 = present_rdk_once(win, roi, coh_second, vf_second, dir_deg, field_rad)
    if a2: abort = True
    if abort: break

    # =======================
    #  応答① Direction?（先に聞く）
    # =======================
    event.clearEvents()
    dir_key = None; dir_choice = None; dir_choice_deg = None; dir_rt = None
    dir_clock = core.Clock()
    while True:
        if event.getKeys(['escape']):
            abort = True
            break
        direction_label.draw(); win.flip()
        keys = event.getKeys(keyList=['up','down'], timeStamped=dir_clock)
        if keys:
            dir_key, dir_rt = keys[0]
            if dir_key == 'up':
                dir_choice, dir_choice_deg = 'up', 90
            else:
                dir_choice, dir_choice_deg = 'down', 270
            break
        if (RESP_TIMEOUT is not None) and (dir_clock.getTime() > RESP_TIMEOUT):
            break
    if abort: break

    # =======================
    #  応答② More Coherent? 
    # =======================
    event.clearEvents()
    interval_key = None; interval_choice = None; interval_rt = None
    resp_clock = core.Clock()
    while True:
        if event.getKeys(['escape']):  
            abort = True
            break
        coh_label.draw(); win.flip()  
        keys = event.getKeys(keyList=['left','right'], timeStamped=resp_clock)
        if keys:
            interval_key, interval_rt = keys[0]
            interval_choice = 1 if interval_key == 'left' else 2
            break
        if (RESP_TIMEOUT is not None) and (resp_clock.getTime() > RESP_TIMEOUT):
            break
    if abort: break


    # ---- 記録 ----
    rows_out.append({
        'participant_id': participant_id,
        'block_name': block_name,
        'trial_index': t_idx,
        'dir_deg': int(dir_deg),
        'coh_center': float(coh_center),
        'coh_periph': float(coh_periph),
        'coh_first': float(coh_first),
        'vf_first': int(vf_first),
        'coh_second': float(coh_second),
        'vf_second': int(vf_second),
        'first_is_center': int(first_is_center),

        # Direction 応答
        'dir_resp_key': dir_key if dir_key else '',
        'dir_resp': '' if dir_choice is None else dir_choice,      # 'up' or 'down'
        'dir_resp_deg': '' if dir_choice_deg is None else int(dir_choice_deg),  # 90 or 270
        'dir_rt_sec': dir_rt if dir_rt is not None else '',

        # 間隔（Coherence）応答
        'interval_choice_key': interval_key if interval_key else '',
        'interval_choice': '' if interval_choice is None else int(interval_choice),  # 1/2
        'interval_rt_sec': interval_rt if interval_rt is not None else '',

        'n_dots': N_DOTS, 'speed': SPEED, 'dot_life': DOT_LIFE,
        'field_diam_used': field_diam,
        'outframe_first': int(out1),
        'outframe_second': int(out2)
    })

    # ---- ITI ----
    iti_clock = core.Clock()
    while iti_clock.getTime() < ITI_SEC:
        if event.getKeys(['escape']):
            abort = True
            break
        fix.draw(); win.flip()
    if abort: break

# ===== 保存・終了 =====
try:
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows_out)
    print(f"Saved: {fname}")
except Exception as e:
    print(f"保存に失敗しました: {e}")

print("Estimated FPS:", win.fps())
win.close(); core.quit()
