#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os, datetime, csv
import numpy as np
from pathlib import Path

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# ===== 追加：ローパス用（deg→px & ぼかしバックエンド） =====
from psychopy.tools.monitorunittools import deg2pix
try:
    import cv2
    _USE_CV2 = True
except Exception:
    _USE_CV2 = False
    from scipy.ndimage import gaussian_filter  # フォールバック

# ===== 実験固定パラメータ =====
N_DOTS          = 120
SPEED           = 0.01
DOT_LIFE        = 3
GAUSS_SIZE      = 0.005
PRE_STIM_FIX_SEC= 1.0
STIM_SEC        = 2.0
ITI_SEC         = 0.5
RESP_TIMEOUT    = None   # 例: 3.0 で回答画面のタイムアウト, Noneで無制限
field_diam = 0.15  
field_rad  = field_diam / 2.0

# ===== 条件CSV 読み込み（(42,3)=coh,dir,vf 期待） =====
def default_cond_matrix():
    coh_levels = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    dirs = [90, 270]
    vfs  = [0, 10, 1000]
    rows = []
    for vf in vfs:
        for d in dirs:
            for c in coh_levels:
                rows.append([c, d, vf])
    return np.array(rows, dtype=float)  # (42,3)

def load_conditions_csv(csv_path):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"条件CSVが見つかりません: {p}")
    # まずpandasで試行（列名あり想定）
    try:
        import pandas as pd
        df = pd.read_csv(p, encoding="utf-8-sig")
        req = ["coherency_cond","direction_cond","vf_cond"]
        if all(c in df.columns for c in req):
            mat = df[req].to_numpy(dtype=float)
            if mat.shape != (42,3):
                raise ValueError(f"形状が{mat.shape}（期待 (42,3)）")
            return mat
        if df.shape[1] >= 3:
            mat = df.iloc[:, :3].to_numpy(dtype=float)
            if mat.shape == (42,3): return mat
            if mat.shape == (3,42): return mat.T
    except Exception:
        pass
    # ヘッダ無しの素朴読み
    rows = []
    with open(p, newline='', encoding='utf-8-sig') as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or not any(cell.strip() for cell in r): continue
            try:
                rows.append([float(r[0]), float(r[1]), float(r[2])])
            except Exception:
                continue
    arr = np.array(rows, dtype=float)
    if arr.shape == (42,3): return arr
    if arr.shape == (3,42): return arr.T
    if arr.shape[0] >= 43 and arr.shape[1] >= 3:
        maybe = arr[1:43, :3]
        if maybe.shape == (42,3): return maybe
    raise ValueError(f"CSVを(42,3)に解釈できませんでした: {arr.shape}")

# ===== 参加者情報の入力（GUI） =====
dlg = gui.Dlg(title="Experiment Info")
dlg.addField("Participant ID:", "")
dlg.addField("Block Name:", "")
ok = dlg.show()
if not ok: core.quit()
participant_id = str(dlg.data[0]).strip() or "unknown"
block_name     = str(dlg.data[1]).strip() or "block"

def sanitize(s):
    s = s.strip().replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_"))

pid_safe   = sanitize(participant_id)
block_safe = sanitize(block_name)

# ===== 条件読み込み & ランダム化 =====
CSV_PATH = "overcoherency_exp1_conds.csv"
try:
    COND_MATRIX = load_conditions_csv(CSV_PATH)
except Exception as e:
    print(f"[WARN] CSV読込に失敗: {e}\n → 既定42条件を使用します。")
    COND_MATRIX = default_cond_matrix()

SHUFFLE_TRIALS = True
SEED = None   # 例: 20250820 にすると順序固定
if SHUFFLE_TRIALS:
    if SEED is not None:
        np.random.seed(SEED)
    idx = np.random.permutation(COND_MATRIX.shape[0])
    COND_MATRIX = COND_MATRIX[idx, :]

N_TRIALS = COND_MATRIX.shape[0]  # ふつうは42

# ===== 画面と刺激の用意 =====
win = visual.Window(
    fullscr=True,            # ★フルスクリーン
    units='height',          # ★縦基準。円や丸マスクが歪みにくい
    color=-1,
    allowGUI=False,
    allowStencil=True,
    winType='pyglet',
    monitor='2b2_monitor'  
)

fix = visual.TextStim(win, text='+', pos=(0,0), color=1.0, height=0.08)
resp_label = visual.TextStim(win, text='Up / Down', pos=(0,0), color=1.0, height=0.1)
coh_label = visual.TextStim(
    win,
    text='Closer to 0% / 100%',
    pos=(0, 0), color=1.0, height=0.08
)
# 画面上に常時出す Coherence 表示（ドットと同時に描く用）
coh_onstim = visual.TextStim(
    win, text='', pos=(0, 0.2), color=1.0, height=0.06  # heightはお好みで
)

# ===== 追加：ドット層表示用の ImageStim（1回だけ作る） =====
imgStim = visual.ImageStim(
    win, units='pix', size=win.size, interpolate=True
)

# ===== データ保存の準備 =====
os.makedirs("data", exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fname = f"data/RDK_{pid_safe}_{block_safe}_{timestamp}.csv"
fname_eye = f"data/RDK_eye_{pid_safe}_{block_safe}_{timestamp}.csv"

thisExp = data.ExperimentHandler(name='lowpass_dots_tracking', version='',
    extraInfo=None, runtimeInfo=None,
    originPath='C:\\Users\\CogInf\\Desktop\\202507_okubo\\lowpass_dots_tracking.py',
    savePickle=True, saveWideText=True,
    dataFileName=fname)
# save a log file for detail verbose info
logFile = logging.LogFile(fname+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

frameTolerance = 0.001  # how close to onset before 'same' frame

fieldnames = [
    'participant_id','block_name','trial',
    'coherency_cond','direction_cond','vf_cond',
    'response_key','response_deg','correct','rt_sec_from_resp_screen',
    'n_dots','speed','dot_life','sigma','field_diam_used',
    'outframe',
    'coh_resp_key', 'coh_resp_choice', 'coh_rt_sec_from_resp_screen'  # ← 追加
]


rows_out = []
# --- Setup input devices ---
ioConfig = {}

# Setup eyetracking
ioConfig['eyetracker.hw.tobii.EyeTracker'] = {
    'name': 'tracker',
    'model_name': '',
    'serial_number': '',
    'runtime_settings': {
        'sampling_rate': 60.0,
    }
}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioServer = io.launchHubServer(window=win, experiment_code='onlyeyetracking', datastore_name=fname_eye, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "fixate" ---
startRecording = hardware.eyetracker.EyetrackerControl(
    tracker=eyetracker,
    actionType='Start Only'
)

# --- Initialize components for Routine "first" ---
roi = visual.ROI(win, name='roi', device=eyetracker,
    debug=False,
    shape='circle',
    pos=[0,0], size=(0.3, 0.3), anchor='center', ori=0.0)

# --- Initialize components for Routine "stopRecord" ---
stopRecording = hardware.eyetracker.EyetrackerControl(
    tracker=eyetracker,
    actionType='Stop Only'
)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 
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
# clear any keypresses from during calibration so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "calibration" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ===== 追加：Eごとのローパス強度（必要なら調整） =====
def sigma_deg_for_vf(vf):
    # 周辺(10°)だけローパス（A=0.1相当で σ≃0.033°）。中心/無制限は0
    if int(vf) == 1000:
        return 0.033
    else:
        return 0.0

# ===== 実験ループ =====
abort = False
def random_points_in_circle(n, radius):
    rho   = radius * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    return np.column_stack([rho*np.cos(theta), rho*np.sin(theta)])

for t_idx in range(N_TRIALS):
    coherence, DIR_DEG, vf_cond = COND_MATRIX[t_idx]

    # ドットの見かけの直径（視認性のためのパラメータ；ローパスとは独立）

    # VF=10°は左右どちらかに寄せる
    if vf_cond == 10:
        offset = np.random.choice([-0.35, 0.35])  
    else:
        offset = 0.0
    dodge = np.array([offset, 0.0], dtype=float)

    # --- RDK状態の初期化 ---
    pos = random_points_in_circle(N_DOTS, field_rad)
    n_signal = int(round(coherence * N_DOTS))
    is_signal = np.zeros(N_DOTS, dtype=bool); is_signal[:n_signal] = True
    np.random.shuffle(is_signal)
    lifetimes = np.random.randint(1, DOT_LIFE + 1, size=N_DOTS)
    step_vec = SPEED * np.array([np.cos(np.deg2rad(DIR_DEG)), np.sin(np.deg2rad(DIR_DEG))])

    fixateComponents = [startRecording]
    for thisComponent in fixateComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # 前固視
    event.clearEvents()
    pre_clock = core.Clock()
    while pre_clock.getTime() < PRE_STIM_FIX_SEC:
        if event.getKeys(['escape']): abort=True; break
        fix.draw(); win.flip()
        
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1
   
        # *startRecording* updates
        if startRecording.status == NOT_STARTED and t >= 0.0-frameTolerance:
            startRecording.frameNStart = frameN
            startRecording.tStart = t
            startRecording.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(startRecording, 'tStartRefresh')
            thisExp.addData('startRecording.started', t)
            startRecording.status = STARTED
        if startRecording.status == STARTED:
            if tThisFlipGlobal > startRecording.tStartRefresh + 0-frameTolerance:
                startRecording.tStop = t
                startRecording.frameNStop = frameN
                thisExp.addData('startRecording.stopped', t)
                startRecording.status = FINISHED
    
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
    
        if not True:
            routineForceEnded = True
            break
        continueRoutine = False
        for thisComponent in fixateComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break
    
    if abort: break
    for thisComponent in fixateComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    if startRecording.status != FINISHED:
        startRecording.status = FINISHED
    routineTimer.reset()

    # 刺激提示（反応は取らない）
    stim_clock = core.Clock()
    event.clearEvents()
    
    # --- Prepare to start Routine "first" ---
    continueRoutine = True
    routineForceEnded = False
    outframe = 0

    roi.setPos((0, 0))
    roi.reset()
    firstComponents = [roi]
    for thisComponent in firstComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    drawn = 0
    # while stim_clock.getTime() < STIM_SEC:
    while drawn <= 18:
        if event.getKeys(['escape']): abort=True; break
            
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1
        # ROI bookkeeping
        if roi.isLookedIn:
            pass
        else:
            outframe += 1
        
        if roi.status == NOT_STARTED and frameN >= 0:
            roi.frameNStart = frameN
            roi.tStart = t
            roi.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(roi, 'tStartRefresh')
            thisExp.timestampOnFlip(win, 'roi.started')
            roi.status = STARTED
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
        else:
            roi.clock.reset()
            roi.wasLookedIn = False
        if roi.status == STARTED:
            if frameN >= (roi.frameNStart + 18):
                roi.tStop = t
                roi.frameNStop = frameN
                thisExp.timestampOnFlip(win, 'roi.stopped')
                roi.status = FINISHED
    
        # ====== RDKの更新 ======
        lifetimes -= 1
        dead = lifetimes <= 0
        if np.any(dead):
            pos[dead] = random_points_in_circle(dead.sum(), field_rad)
            lifetimes[dead] = DOT_LIFE

        pos[is_signal] += step_vec
        idx_noise = ~is_signal
        if np.any(idx_noise):
            phi = 2*np.pi*np.random.rand(idx_noise.sum())
            pos[idx_noise] += SPEED * np.column_stack([np.cos(phi), np.sin(phi)])

        r = np.hypot(pos[:,0], pos[:,1])
        outside = r > field_rad
        if np.any(outside):
            theta = np.arctan2(pos[outside,1], pos[outside,0]) + np.pi
            rr = field_rad * 0.999
            pos[outside] = np.column_stack([rr*np.cos(theta), rr*np.sin(theta)])
            lifetimes[outside] = DOT_LIFE
        
        # ====== ドット層だけをピクセルキャンバスに描画 → ローパス ======
        H, W = win.size[1], win.size[0]
        canvas = np.zeros((H, W), dtype=np.uint8)  # グレイスケール

        # 'height'座標 → ピクセル座標に変換（y軸反転に注意）
        xy_h = pos + dodge  # (N,2)
        xs = np.clip((W/2 + xy_h[:,0]*H).astype(np.int32), 0, W-1)
        ys = np.clip((H/2 - xy_h[:,1]*H).astype(np.int32), 0, H-1)

        # ドット半径[px]（GAUSS_SIZE は直径[height]）
        dot_rad_px = max(1, int(round(GAUSS_SIZE * H / 2.0)))

        # ドットを描画
        if _USE_CV2:
            for x, y in zip(xs, ys):
                cv2.circle(canvas, (int(x), int(y)), dot_rad_px, 255, -1, lineType=cv2.LINE_AA)
        else:
            yy, xx = np.ogrid[:H, :W]
            for x, y in zip(xs, ys):
                mask = (xx-x)**2 + (yy-y)**2 <= dot_rad_px**2
                canvas[mask] = 255

        # 円形アパーチャ（ドット層のみ）
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

        # ローパス（vf_condに応じてσ[deg]→σ[px]）
        sigma_deg = sigma_deg_for_vf(vf_cond)
        sigma_px  = float(deg2pix(sigma_deg, win.monitor)) if sigma_deg > 0 else 0.0

        if (_USE_CV2 and sigma_px >= 0.5):
            blurred = cv2.GaussianBlur(canvas, ksize=(0,0), sigmaX=sigma_px, sigmaY=sigma_px)
        elif ((not _USE_CV2) and sigma_px >= 0.5):
            blurred = gaussian_filter(canvas, sigma=sigma_px, mode='nearest')
        else:
            blurred = canvas  # σ≒0 は無加工

        # 画像として描画（ドット層のみブラー済み）。固視などは非ブラーで上描き
        imgStim.image = np.dstack([blurred]*3).astype(np.float32) / 255.0
        imgStim.draw()
        fix.draw()
        win.flip()
        drawn += 1

    if abort: break
        
    # --- Ending Routine "first" ---
    for thisComponent in firstComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('roi.numLooks', roi.numLooks)
    if roi.numLooks:
       thisExp.addData('roi.timesOn', roi.timesOn)
       thisExp.addData('roi.timesOff', roi.timesOff)
    else:
       thisExp.addData('roi.timesOn', "")
       thisExp.addData('roi.timesOff', "")
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.300000)

    # 回答画面（固視消して「Up / Down」）
    event.clearEvents()
    resp_key = None; resp_deg = None; rt = None
    resp_clock = core.Clock()
    while True:
        if event.getKeys(['escape']): abort=True; break
        resp_label.draw(); win.flip()
        keys = event.getKeys(keyList=['up','down'], timeStamped=resp_clock)
        if keys:
            resp_key, rt = keys[0]
            resp_deg = 90 if resp_key == 'up' else 270
            break  
        if (RESP_TIMEOUT is not None) and (resp_clock.getTime() > RESP_TIMEOUT):
            break
    if abort: break

    correct = (resp_deg == DIR_DEG) if (resp_deg is not None) else False
    
    # ===== 2つ目の回答：コヒーレンスが 0% と 100% のどちらに近い？ =====
    event.clearEvents()
    coh_resp_key = None; coh_resp_choice = None; coh_rt2 = None
    coh_clock = core.Clock()
    while True:
        if event.getKeys(['escape']): abort=True; break
        coh_label.draw(); win.flip()
        keys = event.getKeys(keyList=['left','right'], timeStamped=coh_clock)
        if keys:
            coh_resp_key, coh_rt2 = keys[0]
            coh_resp_choice = 0 if coh_resp_key == 'left' else 100
            break
        if (RESP_TIMEOUT is not None) and (coh_clock.getTime() > RESP_TIMEOUT):
            break
    if abort: break


    rows_out.append({
        'participant_id': participant_id,  
        'block_name': block_name,
        'trial': t_idx+1,
        'coherency_cond': float(coherence),
        'direction_cond': int(DIR_DEG),
        'vf_cond': int(vf_cond),
        'response_key': resp_key if resp_key else '',
        'response_deg': resp_deg if resp_deg is not None else '',
        'correct': int(bool(correct)),
        'rt_sec_from_resp_screen': rt if rt is not None else '',
        'n_dots'  : N_DOTS, 'speed': SPEED, 'dot_life': DOT_LIFE,
        'sigma': sigma_deg  ,
        'field_diam_used': field_diam,
        'outframe': int(outframe),  
        'coh_resp_key': coh_resp_key if coh_resp_key else '',
        'coh_resp_choice': '' if coh_resp_choice is None else int(coh_resp_choice),  # 0 or 100
        'coh_rt_sec_from_resp_screen': coh_rt2 if coh_rt2 is not None else '',
    })

    # ITI
    iti_clock = core.Clock()
    while iti_clock.getTime() < ITI_SEC:
        if event.getKeys(['escape']): abort=True; break
        fix.draw(); win.flip()
    if abort: break

# ===== 保存 & 終了 =====
os.makedirs("data", exist_ok=True)
try:
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows_out)
    print(f"Saved: {fname}")
except Exception as e:
    print(f"保存に失敗しました: {e}")

print("Estimated FPS:", win.fps())
win.close(); core.quit()
  