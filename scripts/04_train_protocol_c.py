#!/usr/bin/env python3
"""
BharatPotHole v2 - YOLO family study: 5 models x 3 seeds, fault-tolerant and resumable
=====================================================================================
Models : YOLO26n, YOLO26s, YOLO26m (primary family), YOLOv8s (paper parity), YOLO11s (predecessor)
Seeds  : 0, 1, 2          Data: BharatPotHole v2 video-disjoint split (protocol C)
Recipe : each model's Ultralytics default recipe (optimizer='auto'), identical data/epochs/imgsz/augmentation

HOW TO RUN (Colab)
    Cell 1:  from google.colab import drive; drive.mount('/content/drive')
    Cell 2:  !nvidia-smi
    Cell 3:  !python "/content/drive/MyDrive/POTHOLE/scripts/train_family_study.py"
Re-run Cell 1 + Cell 3 after ANY interruption (disconnect, power cut, crash, Ctrl-C).
It continues exactly where it stopped. Other flags:
    --summary-only      rebuild runs_summary.csv / summary.md from Drive and exit
    --retry-failed      also retry runs previously marked 'failed'
    --force-unlock      ignore a lock held by another (dead) session
    --accept-config-change   allow changed training settings for runs not yet started

FAULT TOLERANCE
  * Supervisor/worker design: every run trains in a fresh child process, so a CUDA OOM or a
    corrupted CUDA context never poisons the next run.
  * Training writes to fast local disk; after EVERY epoch the checkpoint is copied to Drive
    atomically (copy to .tmp, then rename) with a rolling previous copy (last_prev.pt). A power
    cut during a copy therefore cannot destroy the last good checkpoint.
  * On restart, the newest *loadable* checkpoint (local, Drive last.pt, Drive last_prev.pt) is
    restored, paths/batch are patched inside it, results.csv is trimmed to that epoch, and
    Ultralytics resumes with optimizer, EMA, epoch counter and best-fitness intact.
  * GPU OOM  -> batch halved (min 2) and the run resumes. Ultralytics accumulates gradients to a
    nominal batch of 64 (nbs), so the effective batch - and the optimisation - stays comparable.
    Host-RAM kills -> dataloader workers reduced. Other errors -> up to 3 attempts, then the run
    is marked failed and the queue moves on.
  * Every event (resume, OOM, batch change, GPU used, errors) is logged in state.json and
    events.jsonl, so interruptions can be reported transparently in the paper.
  * The Ultralytics version is pinned at the first run and re-installed identically later.
  * A lock file prevents two sessions from training the same experiment simultaneously.

OUTPUT ON DRIVE  (/content/drive/MyDrive/POTHOLE/experiments/family_study_v2/)
  runs/<model>_s<seed>/weights/best.pt        best checkpoint (use for later evaluation)
  runs/<model>_s<seed>/weights/last.pt        final checkpoint
  runs/<model>_s<seed>/weights/last_full.pt   final checkpoint WITH optimizer (to extend training)
  runs/<model>_s<seed>/results.csv, args.yaml, plots
  state.json, events.jsonl, logs/, pretrained/ (exact starting weights + MD5)
  runs_summary.csv, summary.md  (VALIDATION metrics only; test evaluation is a separate step)
"""
import os, sys, json, time, socket, shutil, argparse, subprocess, hashlib, traceback, zipfile, signal
from pathlib import Path
from datetime import datetime, timezone

# ============================== CONFIGURATION =========================================
IN_COLAB = Path('/content').exists()
DRIVE_ROOT = Path(os.environ.get('POTHOLE_DRIVE', '/content/drive/MyDrive/POTHOLE'))
EXP_NAME = os.environ.get('EXP_NAME', 'family_study_v2')
DRIVE_EXP = DRIVE_ROOT / 'experiments' / EXP_NAME
WORK_DIR = Path(os.environ.get('WORK_DIR', '/content/work' if IN_COLAB else str(Path.home() / 'pothole_work'))) / EXP_NAME
V2_ZIP = DRIVE_ROOT / 'BharatPotHole_v2' / 'BharatPotHole_v2.zip'
V2_LOCAL = Path(os.environ.get('V2_LOCAL', '/content/BharatPotHole_v2' if IN_COLAB else str(Path.home() / 'BharatPotHole_v2')))

MODELS = os.environ.get('MODELS', 'yolo26n.pt,yolo26s.pt,yolo26m.pt,yolov8s.pt,yolo11s.pt').split(',')
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
TRAIN_ARGS = dict(
    epochs=int(os.environ.get('EPOCHS', 100)), patience=int(os.environ.get('PATIENCE', 30)),
    imgsz=int(os.environ.get('IMGSZ', 640)), batch=int(os.environ.get('BATCH', 16)),
    optimizer='auto',          # each model's default recipe (YOLO26 keeps its own optimizer logic)
    pretrained=True, deterministic=True, amp=True, cache=False,
    plots=True, val=True, save=True, save_period=-1, exist_ok=True, verbose=True,
)
DEFAULT_WORKERS = min(8, os.cpu_count() or 2)
MIN_BATCH = 2
MAX_ATTEMPTS = 3              # non-OOM failures allowed per run
LOCK_STALE_MIN = 20           # a lock without heartbeat for this long is considered dead
ALLOW_CPU = os.environ.get('ALLOW_CPU') == '1'     # testing only
EXIT_OK, EXIT_ERR, EXIT_OOM, EXIT_DONE_BEFORE, EXIT_CUDA_FATAL, EXIT_HOST_RAM = 0, 1, 3, 4, 5, 6

STATE_F, EVENTS_F, LOCK_F = DRIVE_EXP / 'state.json', DRIVE_EXP / 'events.jsonl', DRIVE_EXP / 'lock.json'
LOCAL_STATE_F = WORK_DIR / 'state.json'
HOST = f'{socket.gethostname()}:{os.getpid()}'


# ============================== SMALL UTILITIES =======================================
def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def atomic_copy(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def atomic_write_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def load_state():
    for p in (STATE_F, LOCAL_STATE_F):
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception as e:
            log(f'WARNING: could not read {p}: {e}')
    return None

def save_state(state):
    txt = json.dumps(state, indent=2, default=str)
    atomic_write_text(LOCAL_STATE_F, txt)
    try:
        atomic_write_text(STATE_F, txt)
    except Exception as e:
        log(f'WARNING: Drive state write failed ({e}); local copy kept')

def update_run(run_id, **kw):
    st = load_state()
    r = st['runs'][run_id]
    add_sec = kw.pop('train_seconds_add', 0)
    r['train_seconds'] = r.get('train_seconds', 0) + add_sec
    r.update(kw)
    save_state(st)
    return st

def event(run_id, kind, **kw):
    rec = {'time': now(), 'host': HOST, 'run': run_id, 'event': kind, **kw}
    line = json.dumps(rec, default=str) + '\n'
    for p in (WORK_DIR / 'events.jsonl', EVENTS_F):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'a') as f:
                f.write(line)
        except Exception:
            pass

def gpu_info():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else 'none'
    except Exception:
        return 'none'

def heartbeat():
    try:
        atomic_write_text(LOCK_F, json.dumps({'host': HOST, 'heartbeat': time.time(), 'iso': now()}))
    except Exception:
        pass


# ============================== CHECKPOINT HANDLING ===================================
def ckpt_info(path):
    """Return dict(epoch, epochs, has_opt, finished) or None if the file is unreadable."""
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    ep = ck.get('epoch', -1)
    ta = ck.get('train_args', {}) or {}
    epochs = int(ta.get('epochs', TRAIN_ARGS['epochs']))
    has_opt = ck.get('optimizer') is not None
    finished = ep == -1 or not has_opt or ep + 1 >= epochs
    return {'epoch': ep, 'epochs': epochs, 'has_opt': has_opt, 'finished': finished}

def patch_ckpt(path, **train_arg_updates):
    """Rewrite paths/batch/device inside a checkpoint so resume works on any machine/session."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ta = dict(ck.get('train_args', {}) or {})
    ta.update(train_arg_updates)
    ck['train_args'] = ta
    tmp = Path(str(path) + '.tmp')
    torch.save(ck, tmp)
    os.replace(tmp, path)

def trim_results_csv(csv_path, last_epoch_1based):
    if not csv_path.exists():
        return
    lines = csv_path.read_text().splitlines()
    if len(lines) < 2:
        return
    keep = [lines[0]]
    for ln in lines[1:]:
        try:
            if int(float(ln.split(',')[0].strip())) <= last_epoch_1based:
                keep.append(ln)
        except ValueError:
            pass
    if len(keep) != len(lines):
        atomic_write_text(csv_path, '\n'.join(keep) + '\n')

def restore_checkpoint(local_run, drive_run):
    """Pick the newest loadable checkpoint among local/Drive copies and place it locally."""
    wl, wd = local_run / 'weights', drive_run / 'weights'
    wl.mkdir(parents=True, exist_ok=True)
    cands = [('local', wl / 'last.pt'), ('drive', wd / 'last.pt'), ('drive_prev', wd / 'last_prev.pt')]
    best_src, best_info = None, None
    for tag, p in cands:
        if p.exists():
            info = ckpt_info(p)
            if info is None:
                log(f'  {tag} checkpoint unreadable (likely interrupted write) -> skipped: {p}')
                continue
            rank = 10 ** 6 if info['finished'] and info['epoch'] == -1 else info['epoch']
            if best_info is None or rank > best_info['_rank']:
                best_src, best_info = (tag, p), {**info, '_rank': rank}
    if best_src is None:
        return None, None
    tag, p = best_src
    if p != wl / 'last.pt':
        atomic_copy(p, wl / 'last.pt')
    for name in ('best.pt',):
        if not (wl / name).exists() and (wd / name).exists() and ckpt_info(wd / name) is not None:
            atomic_copy(wd / name, wl / name)
    for name in ('results.csv', 'args.yaml'):
        src = drive_run / name
        if src.exists() and (not (local_run / name).exists() or
                             src.stat().st_size > (local_run / name).stat().st_size):
            atomic_copy(src, local_run / name)
    if best_info['epoch'] >= 0:
        trim_results_csv(local_run / 'results.csv', best_info['epoch'] + 1)
    log(f'  restored checkpoint from {tag}: epoch {best_info["epoch"] + 1 if best_info["epoch"] >= 0 else "final"}')
    return wl / 'last.pt', best_info

def sync_run(local_run, drive_run, final=False, _seen={}):
    """Copy the local run to Drive safely. Called after every epoch and at the end."""
    wl, wd = local_run / 'weights', drive_run / 'weights'
    wd.mkdir(parents=True, exist_ok=True)
    last, best = wl / 'last.pt', wl / 'best.pt'
    if final:
        dl = wd / 'last.pt'
        if dl.exists():
            info = ckpt_info(dl)
            if info and info['has_opt']:
                os.replace(dl, wd / 'last_full.pt')      # keep full final ckpt (with optimizer)
        if last.exists():
            atomic_copy(last, wd / 'last.pt')
        if best.exists():
            atomic_copy(best, wd / 'best.pt')
        if (wd / 'last_prev.pt').exists():
            (wd / 'last_prev.pt').unlink()
        for f in local_run.iterdir():                 # plots, csv, yaml
            if f.is_file() and f.suffix.lower() in ('.csv', '.yaml', '.png', '.jpg', '.txt', '.json'):
                atomic_copy(f, drive_run / f.name)
        return
    if last.exists():
        if (wd / 'last.pt').exists():
            os.replace(wd / 'last.pt', wd / 'last_prev.pt')
        atomic_copy(last, wd / 'last.pt')
    if best.exists():
        sig = (best.stat().st_mtime, best.stat().st_size)
        if _seen.get(str(best)) != sig or not (wd / 'best.pt').exists():
            atomic_copy(best, wd / 'best.pt'); _seen[str(best)] = sig
    for name in ('results.csv', 'args.yaml'):
        if (local_run / name).exists():
            atomic_copy(local_run / name, drive_run / name)


# ============================== WORKER (one training run) =============================
def worker(run_id):
    import torch
    from ultralytics import YOLO
    st = load_state()
    r = st['runs'][run_id]
    local_run, drive_run = WORK_DIR / 'runs' / run_id, DRIVE_EXP / 'runs' / run_id
    data_yaml = WORK_DIR / 'data_v2.yaml'
    device = 0 if torch.cuda.is_available() else 'cpu'
    timer = {'t': time.time()}

    def on_model_save(trainer):
        try:
            sync_run(Path(trainer.save_dir), drive_run)
        except Exception as e:
            log(f'WARNING: Drive sync failed ({e}); training continues, local checkpoint kept')
        try:
            t = time.time(); dt = t - timer['t']; timer['t'] = t
            update_run(run_id, last_epoch=int(trainer.epoch) + 1, best_fitness=float(trainer.best_fitness or 0),
                       batch_in_use=int(trainer.args.batch), train_seconds_add=dt, last_update=now())
            heartbeat()
        except Exception as e:
            log(f'WARNING: state update failed ({e})')

    try:
        ckpt, info = restore_checkpoint(local_run, drive_run)
        if info is not None and info['finished']:
            log(f'{run_id}: training already finished - finalising')
            sync_run(local_run, drive_run, final=True)
            return EXIT_DONE_BEFORE
        common = dict(data=str(data_yaml), project=str(WORK_DIR / 'runs'), name=run_id, exist_ok=True,
                      save_dir=str(local_run), batch=int(r['batch']), workers=int(r['workers']), device=device)
        if ckpt is not None:
            patch_ckpt(ckpt, **common)
            event(run_id, 'resume', from_epoch=info['epoch'] + 1, batch=r['batch'], workers=r['workers'], gpu=gpu_info())
            log(f'{run_id}: RESUMING at epoch {info["epoch"] + 2}/{info["epochs"]} (batch {r["batch"]}, workers {r["workers"]})')
            model = YOLO(str(ckpt))
            model.add_callback('on_model_save', on_model_save)
            model.train(resume=True, batch=int(r['batch']), workers=int(r['workers']), device=device)
        else:
            pre = WORK_DIR / 'pretrained' / r['model']
            event(run_id, 'start', batch=r['batch'], workers=r['workers'], gpu=gpu_info())
            log(f'{run_id}: STARTING from {pre.name} (seed {r["seed"]}, batch {r["batch"]})')
            model = YOLO(str(pre))
            model.add_callback('on_model_save', on_model_save)
            args = {**TRAIN_ARGS, 'batch': int(r['batch'])}
            model.train(data=str(data_yaml), project=str(WORK_DIR / 'runs'), name=run_id, seed=int(r['seed']),
                        workers=int(r['workers']), device=device, **args)
        sync_run(local_run, drive_run, final=True)
        extra = {}
        try:
            best = local_run / 'weights' / 'best.pt'
            from ultralytics.utils.torch_utils import get_num_params, get_flops
            net = YOLO(str(best)).model
            extra = {'params': int(get_num_params(net)),
                     'gflops': round(float(get_flops(net, TRAIN_ARGS['imgsz'])), 2)}
        except Exception:
            pass
        update_run(run_id, **extra)
        return EXIT_OK
    except Exception as e:
        msg = f'{type(e).__name__}: {e}'
        low = msg.lower()
        try:
            sync_run(local_run, drive_run)
        except Exception:
            pass
        tb = traceback.format_exc()
        event(run_id, 'error', message=msg[:2000])
        try:
            st = load_state(); st['runs'][run_id].setdefault('errors', []).append(
                {'time': now(), 'message': msg[:2000], 'traceback': tb[-4000:]}); save_state(st)
        except Exception:
            pass
        print(tb, flush=True)
        if isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ())) or 'out of memory' in low:
            return EXIT_OOM
        if 'killed by signal' in low or 'bus error' in low or 'shared memory' in low or 'cannot allocate memory' in low:
            return EXIT_HOST_RAM
        if 'cuda error' in low or 'cudnn' in low or 'cublas' in low or 'device-side assert' in low:
            return EXIT_CUDA_FATAL
        return EXIT_ERR


# ============================== SUPERVISOR ============================================
def pip_install(spec):
    log(f'pip install {spec}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', spec], check=True)

def installed_version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None

def ensure_environment(state):
    pinned = (state or {}).get('env', {}).get('ultralytics')
    cur = installed_version('ultralytics')
    if pinned and cur != pinned:
        log(f'Ultralytics {cur} installed but experiment pinned to {pinned} -> reinstalling pinned version')
        pip_install(f'ultralytics=={pinned}')
    elif cur is None:
        pip_install('ultralytics')
    os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')    # no CUDA context in supervisor
    import torch
    if torch.version.cuda is None and not ALLOW_CPU:
        g = gpu_info()
        sys.exit('ERROR: CPU-only PyTorch is installed' + (f' although a GPU is present ({g}).\n'
                 '  Fix: pip install --force-reinstall torch torchvision --index-url '
                 'https://download.pytorch.org/whl/cu126 ; restart the session; re-run.' if g != 'none'
                 else '.\n  Fix: Runtime > Change runtime type > GPU, then re-run.'))
    if not torch.cuda.is_available() and not ALLOW_CPU:
        sys.exit('ERROR: no GPU visible. Runtime > Change runtime type > GPU, then re-run.')
    return {'ultralytics': installed_version('ultralytics'), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'python': sys.version.split()[0]}

def ensure_data():
    if not (V2_LOCAL / 'data.yaml').exists():
        assert V2_ZIP.exists(), f'Dataset zip not found: {V2_ZIP}'
        log(f'Extracting {V2_ZIP.name} ...')
        with zipfile.ZipFile(V2_ZIP) as zf:
            zf.extractall(V2_LOCAL.parent)
    import yaml
    d = yaml.safe_load((V2_LOCAL / 'data.yaml').read_text())
    for s in ('train', 'valid'):
        n = len(list((V2_LOCAL / s / 'images').glob('*')))
        assert n > 0, f'No images in {V2_LOCAL / s / "images"}'
    out = WORK_DIR / 'data_v2.yaml'
    atomic_write_text(out, f"path: {V2_LOCAL}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                           f"nc: {d.get('nc', 1)}\nnames: {json.dumps(d.get('names', ['pothole']))}\n")
    return out

def ensure_pretrained(state):
    from ultralytics.utils.downloads import attempt_download_asset
    pd_, pl = DRIVE_EXP / 'pretrained', WORK_DIR / 'pretrained'
    pd_.mkdir(parents=True, exist_ok=True); pl.mkdir(parents=True, exist_ok=True)
    rec = state.setdefault('pretrained', {})
    for m in MODELS:
        if not (pd_ / m).exists():
            cwd = os.getcwd(); os.chdir(pl)
            try:
                p = Path(attempt_download_asset(m))
            finally:
                os.chdir(cwd)
            p = p if p.is_absolute() else pl / p
            assert p.exists(), f'Could not download {m} - is this Ultralytics version new enough?'
            atomic_copy(p, pd_ / m)
        if not (pl / m).exists():
            atomic_copy(pd_ / m, pl / m)
        h = md5(pd_ / m)
        if m in rec and rec[m] != h:
            sys.exit(f'ERROR: pretrained weights for {m} changed (MD5 mismatch) - protocol would not be identical.')
        rec[m] = h
    log('Pretrained weights ready: ' + ', '.join(f'{m} ({h[:8]})' for m, h in rec.items()))

def acquire_lock(force):
    if LOCK_F.exists() and not force:
        try:
            lk = json.loads(LOCK_F.read_text())
            age_min = (time.time() - lk.get('heartbeat', 0)) / 60
            lk_host, _, lk_pid = str(lk.get('host', '')).rpartition(':')
            same_machine_dead = False
            if lk_host == socket.gethostname():          # same machine: take over if that process is gone
                try:
                    os.kill(int(lk_pid), 0)
                except (OSError, ValueError):
                    same_machine_dead = True
            if lk.get('host') != HOST and age_min < LOCK_STALE_MIN and not same_machine_dead:
                sys.exit(f'ERROR: experiment locked by {lk.get("host")} (heartbeat {age_min:.1f} min ago).\n'
                         f'  If that session is dead, wait {LOCK_STALE_MIN} min or use --force-unlock.')
        except (json.JSONDecodeError, OSError):
            pass
    heartbeat()

def release_lock():
    try:
        if LOCK_F.exists() and json.loads(LOCK_F.read_text()).get('host') == HOST:
            LOCK_F.unlink()
    except Exception:
        pass

def init_state(state, env, accept_change):
    cfg = {'models': MODELS, 'seeds': SEEDS, 'train_args': TRAIN_ARGS, 'dataset_zip': str(V2_ZIP)}
    if state is None:
        state = {'created': now(), 'env': env, 'config': cfg, 'runs': {}}
    else:
        old = state.get('config', {}).get('train_args')
        if old and old != TRAIN_ARGS:
            started = [k for k, r in state['runs'].items() if r['status'] != 'pending']
            if started and not accept_change:
                sys.exit(f'ERROR: training settings differ from those used for {len(started)} started runs:\n'
                         f'  stored : {old}\n  current: {TRAIN_ARGS}\n'
                         '  Restore the original settings, or use --accept-config-change (not recommended).')
            state['config_changes'] = state.get('config_changes', []) + [{'time': now(), 'from': old, 'to': TRAIN_ARGS}]
        state['config'] = cfg
        state['env'] = {**env, 'first_ultralytics': state.get('env', {}).get('first_ultralytics', env['ultralytics'])}
    state['env'].setdefault('first_ultralytics', env['ultralytics'])
    for seed in SEEDS:                           # seed-outer order: a complete comparison exists early
        for m in MODELS:
            rid = f'{Path(m).stem}_s{seed}'
            state['runs'].setdefault(rid, {'model': m, 'seed': seed, 'status': 'pending',
                                           'batch': TRAIN_ARGS['batch'], 'workers': DEFAULT_WORKERS,
                                           'attempts': 0, 'oom_events': [], 'sessions': [],
                                           'train_seconds': 0, 'last_epoch': 0})
    for r in state['runs'].values():             # a 'running' run from a dead session is resumable
        if r['status'] == 'running':
            r['status'] = 'pending'
    return state

def run_child(run_id):
    log_path = WORK_DIR / 'logs' / f'{run_id}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None)
    proc = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--worker', run_id],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    last_copy = time.time()
    try:
        with open(log_path, 'ab') as lf:
            lf.write(f'\n===== session {now()} {HOST} =====\n'.encode())
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk); sys.stdout.flush(); lf.write(chunk)
                if time.time() - last_copy > 60:
                    lf.flush()
                    try:
                        atomic_copy(log_path, DRIVE_EXP / 'logs' / log_path.name)
                    except Exception:
                        pass
                    last_copy = time.time()
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        try:
            atomic_copy(log_path, DRIVE_EXP / 'logs' / log_path.name)
        except Exception:
            pass
    return rc

def supervise(retry_failed):
    state = load_state()
    order = list(state['runs'])
    pending = [k for k in order if state['runs'][k]['status'] in ('pending',) or
               (retry_failed and state['runs'][k]['status'] == 'failed')]
    log(f'Queue: {len(pending)} of {len(order)} runs to do -> {pending}')
    for rid in pending:
        while True:
            st = load_state(); r = st['runs'][rid]
            r['status'] = 'running'
            r['sessions'].append({'start': now(), 'host': HOST, 'gpu': gpu_info()})
            r.setdefault('started', now())
            save_state(st); heartbeat()
            log(f'==================== {rid}  (model {r["model"]}, seed {r["seed"]}) ====================')
            rc = run_child(rid)
            st = load_state(); r = st['runs'][rid]
            if rc in (EXIT_OK, EXIT_DONE_BEFORE):
                r['status'] = 'done'; r['finished'] = now()
                atomic_write_text(DRIVE_EXP / 'runs' / rid / 'DONE', now())
                save_state(st); event(rid, 'done'); log(f'{rid}: DONE')
                break
            if rc == EXIT_OOM:
                if r['batch'] <= MIN_BATCH:
                    r['status'] = 'failed'; save_state(st); event(rid, 'failed', reason='OOM at min batch'); break
                nb = max(MIN_BATCH, r['batch'] // 2)
                r['oom_events'].append({'time': now(), 'batch_before': r['batch'], 'batch_after': nb,
                                        'epoch': r.get('last_epoch'), 'gpu': gpu_info()})
                event(rid, 'oom', batch_before=r['batch'], batch_after=nb); log(f'{rid}: GPU OOM -> batch {r["batch"]} -> {nb}')
                r['batch'] = nb; r['status'] = 'pending'; save_state(st)
                continue
            if rc == EXIT_HOST_RAM or rc in (-9, 137):
                if r['workers'] > 0:
                    nw = r['workers'] // 2
                    event(rid, 'host_ram', workers_before=r['workers'], workers_after=nw)
                    log(f'{rid}: host RAM exhausted -> dataloader workers {r["workers"]} -> {nw}')
                    r['workers'] = nw
                elif r['batch'] > MIN_BATCH:
                    r['batch'] = max(MIN_BATCH, r['batch'] // 2)
                    event(rid, 'host_ram_batch', batch_after=r['batch'])
                else:
                    r['status'] = 'failed'; save_state(st); event(rid, 'failed', reason='host RAM'); break
                r['status'] = 'pending'; save_state(st)
                continue
            r['attempts'] = r.get('attempts', 0) + 1
            if r['attempts'] >= MAX_ATTEMPTS:
                r['status'] = 'failed'; save_state(st)
                event(rid, 'failed', reason=f'exit code {rc}', attempts=r['attempts'])
                log(f'{rid}: FAILED after {r["attempts"]} attempts (see state.json errors / logs) - moving on')
                break
            r['status'] = 'pending'; save_state(st)
            log(f'{rid}: error (exit {rc}) - retry {r["attempts"]}/{MAX_ATTEMPTS - 1} in a fresh process')
            time.sleep(10)
        build_summary()


# ============================== SUMMARY ===============================================
def build_summary():
    import pandas as pd
    st = load_state()
    rows = []
    for rid, r in st['runs'].items():
        row = {'run_id': rid, 'model': Path(r['model']).stem, 'seed': r['seed'], 'status': r['status'],
               'epochs_completed': r.get('last_epoch', 0), 'batch_final': r['batch'], 'workers_final': r['workers'],
               'oom_events': len(r.get('oom_events', [])), 'sessions': len(r.get('sessions', [])),
               'train_hours': round(r.get('train_seconds', 0) / 3600, 2),
               'params': r.get('params'), 'gflops': r.get('gflops'),
               'gpus': '; '.join(sorted({s.get('gpu', '') for s in r.get('sessions', [])}))}
        res = DRIVE_EXP / 'runs' / rid / 'results.csv'
        if res.exists():
            try:
                d = pd.read_csv(res); d.columns = [c.strip() for c in d.columns]
                k = 'metrics/mAP50-95(B)'
                if k in d and len(d):
                    b = d.loc[d[k].idxmax()]
                    row.update({'epochs_completed': int(d['epoch'].max()), 'best_epoch': int(b['epoch']),
                                'val_precision': round(float(b['metrics/precision(B)']), 4),
                                'val_recall': round(float(b['metrics/recall(B)']), 4),
                                'val_mAP50': round(float(b['metrics/mAP50(B)']), 4),
                                'val_mAP50_95': round(float(b[k]), 4)})
            except Exception as e:
                row['summary_error'] = str(e)
        row['best_pt'] = str(DRIVE_EXP / 'runs' / rid / 'weights' / 'best.pt') \
            if (DRIVE_EXP / 'runs' / rid / 'weights' / 'best.pt').exists() else ''
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(DRIVE_EXP / 'runs_summary.csv', index=False)
    lines = [f'# Family study - training summary ({now()})',
             f'Ultralytics {st["env"].get("ultralytics")} | torch {st["env"].get("torch")} | settings {json.dumps(TRAIN_ARGS)}',
             '', 'VALIDATION metrics at the best epoch (test-set evaluation is a separate step).', '']
    done = s[s.status == 'done']
    if len(done) and 'val_mAP50_95' in done:
        g = done.groupby('model')[['val_precision', 'val_recall', 'val_mAP50', 'val_mAP50_95', 'train_hours']]
        agg = g.agg(['mean', 'std', 'count']).round(4)
        lines += ['## Mean / std over completed seeds', '```', agg.to_string(), '```', '']
    lines += ['## All runs', '```', s.drop(columns=['best_pt']).to_string(index=False), '```']
    oom = s[s.oom_events > 0]
    if len(oom):
        lines += ['', f'Note: {len(oom)} run(s) had batch size reduced after GPU OOM; gradient accumulation to '
                  'nbs=64 keeps the effective batch unchanged. Details in state.json.']
    atomic_write_text(DRIVE_EXP / 'summary.md', '\n'.join(lines) + '\n')
    counts = s.status.value_counts().to_dict()
    log(f'Summary updated: {counts} -> {DRIVE_EXP / "summary.md"}')


# ============================== ENTRY POINT ===========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker'); ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--retry-failed', action='store_true'); ap.add_argument('--force-unlock', action='store_true')
    ap.add_argument('--accept-config-change', action='store_true')
    a, _unknown = ap.parse_known_args()          # tolerate kernel args (e.g. Colab's -f kernel.json)
    if '__file__' not in globals() or 'ipykernel' in sys.modules:
        sys.exit('This script must run as a .py file, not inside a notebook cell.\n'
                 '  1) In a cell whose FIRST line is:\n'
                 '       %%writefile /content/drive/MyDrive/POTHOLE/scripts/train_family_study.py\n'
                 '     paste the whole script and run it (or upload the .py file to that folder).\n'
                 '  2) Then run:  !python "/content/drive/MyDrive/POTHOLE/scripts/train_family_study.py"')
    if a.worker:
        sys.exit(worker(a.worker))
    if not DRIVE_ROOT.exists():
        sys.exit(f'ERROR: {DRIVE_ROOT} not found. In Colab run first: '
                 'from google.colab import drive; drive.mount("/content/drive")')
    DRIVE_EXP.mkdir(parents=True, exist_ok=True); WORK_DIR.mkdir(parents=True, exist_ok=True)
    if a.summary_only:
        build_summary(); return
    acquire_lock(a.force_unlock)
    try:
        state = load_state()
        env = ensure_environment(state)
        state = init_state(state, env, a.accept_config_change)
        save_state(state)
        ensure_data()
        ensure_pretrained(state); save_state(state)
        log(f'GPU: {gpu_info()} | Ultralytics {env["ultralytics"]} | torch {env["torch"]} | work dir {WORK_DIR}')
        supervise(a.retry_failed)
        build_summary()
        st = load_state()
        left = [k for k, r in st['runs'].items() if r['status'] != 'done']
        log('ALL RUNS COMPLETE.' if not left else f'Finished this session; not done: {left}')
    except KeyboardInterrupt:
        log('Interrupted by user - progress is saved; re-run to continue.')
        st = load_state()
        if st:
            for r in st['runs'].values():
                if r['status'] == 'running':
                    r['status'] = 'pending'
            save_state(st)
    finally:
        release_lock()


if __name__ == '__main__':
    main()