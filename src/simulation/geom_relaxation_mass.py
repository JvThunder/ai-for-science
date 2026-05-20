#!/usr/bin/env python3
import subprocess, os, time, sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
from datetime import datetime
import re
# ================== PATH SETTINGS ==================
pw_exe     = "/app/apps/qe/7.4-icx24-mkl/bin/pw.x"      # Linux QE binary
pseudo_dir = "pseudo"       # adjust to your pseudopotential path
def get_next_logfile(base_name="scheduler_log.txt"):
    """
    Returns the first available log filename:
    scheduler_log.txt, scheduler_log1.txt, scheduler_log2.txt, ...
    """
    # If "scheduler_log.txt" does not exist → use it
    if not os.path.exists(base_name):
        return base_name
    
    # Otherwise check scheduler_log1.txt, scheduler_log2.txt, ...
    base, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        candidate = f"{base}{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


# ==== Use it here ====
log_file = get_next_logfile("scheduler_log.txt")

# Write the program name as the first line of the log
with open(log_file, "w") as f:
    exe_name = os.path.basename(sys.argv[0])
    f.write(f"=== Log started by {exe_name} ===\n")

print(f"[INFO] Logging to: {log_file}")

# ================== BASE QUALITY SETTINGS ==================
quality_settings = {
    "PBE": {
        "USPP": {
            "1": {"ecutwfc": 45, "ratio": 8,  "kpoints": "gamma"}
        },
        "PAW": {
            "Extralow": {"ecutwfc": 35,  "ratio": 4, "kpoints": "1 1 1 0 0 0"},
            "Low":      {"ecutwfc": 50,  "ratio": 4, "kpoints": "2 2 2 0 0 0"},
            "Medium":   {"ecutwfc": 70,  "ratio": 4, "kpoints": "4 4 4 0 0 0"},
            "High":     {"ecutwfc": 90,  "ratio": 4, "kpoints": "6 6 6 0 0 0"},
            "Extrahigh":{"ecutwfc": 110, "ratio": 4, "kpoints": "8 8 8 0 0 0"}
        },
        "NCPP": {
            "Extralow": {"ecutwfc": 50,  "ratio": 4, "kpoints": "1 1 1 0 0 0"},
            "Low":      {"ecutwfc": 70,  "ratio": 4, "kpoints": "2 2 2 0 0 0"},
            "Medium":   {"ecutwfc": 90,  "ratio": 4, "kpoints": "4 4 4 0 0 0"},
            "High":     {"ecutwfc": 110, "ratio": 4, "kpoints": "6 6 6 0 0 0"},
            "Extrahigh":{"ecutwfc": 130, "ratio": 4, "kpoints": "8 8 8 0 0 0"}
        }
    },
    "LDA": {
        "USPP": {
            "Extralow": {"ecutwfc": 20,  "ratio": 8, "kpoints": "1 1 1 0 0 0"},
            "Low":      {"ecutwfc": 30,  "ratio": 8, "kpoints": "2 2 2 0 0 0"},
            "Medium":   {"ecutwfc": 40,  "ratio": 8, "kpoints": "4 4 4 0 0 0"},
            "High":     {"ecutwfc": 55,  "ratio": 8, "kpoints": "6 6 6 0 0 0"},
            "Extrahigh":{"ecutwfc": 70,  "ratio": 8, "kpoints": "8 8 8 0 0 0"}
        }
    }
}

# ================== FUNCTION TO GET CUTOFFS ==================
def get_cutoffs(xc_family, pseudo_type, quality_label):
    entry = quality_settings[xc_family][pseudo_type][quality_label]
    ecutwfc = entry["ecutwfc"]
    ecutrho = ecutwfc * entry["ratio"]
    return ecutwfc, ecutrho, entry["kpoints"]

# ================== DETECT PSEUDOPOTENTIALS ==================
xc_family = "PBE"
pseudo_type = "USPP"

pseudo_dir_selected = os.path.join(pseudo_dir, xc_family, pseudo_type)
if not os.path.exists(pseudo_dir_selected):
    raise RuntimeError(f"Pseudo directory {pseudo_dir_selected} does not exist")

pseudopotentials = {}
for fname in os.listdir(pseudo_dir_selected):
    if fname.lower().endswith(".upf"):
        element = fname.split(".")[0].capitalize()
        pseudopotentials[element] = {
            "file": os.path.join(pseudo_dir_selected, fname),
            "xc": xc_family,
            "type": pseudo_type
        }

print(f"[INFO] Default XC family enforced: {xc_family}/{pseudo_type}")
print("[INFO] Pseudopotentials collected:")
for el, meta in pseudopotentials.items():
    print(f"  {el} → {os.path.basename(meta['file'])}")

# ================== LOGGING ==================
def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")

# ================== RUNNER ==================
def run_qe_with_timer(exe, infile, outfile, nprocs):
    start = time.time()
    prefix = os.path.splitext(os.path.basename(infile))[0]
    relax_log = os.path.join(os.path.dirname(outfile), f"{prefix}_relax_steps.log")

    with open(outfile, "w") as f, open(relax_log, "w") as flog:
        cmd = ["mpirun", "-np", str(nprocs), exe, "-in", infile]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1)
        capture_cell, capture_atoms = False, False
        tmp = []
        step_counter = 0

        for line in process.stdout:
            f.write(line); f.flush()
            sys.stdout.write(line); sys.stdout.flush()

            if line.strip().startswith("CELL_PARAMETERS"):
                capture_cell = True
                tmp = [line.strip()]
                continue

            if capture_cell and len(tmp) < 4:
                tmp.append(line.strip())
                if len(tmp) == 4:
                    flog.write(f"=== Relaxation Step {step_counter} ===\n")
                    flog.write("\n".join(tmp) + "\n")
                    capture_cell = False
                    continue

            if line.strip().startswith("ATOMIC_POSITIONS"):
                capture_atoms = True
                flog.write("ATOMIC_POSITIONS angstrom\n")
                continue

            if capture_atoms:
                parts = line.split()
                if len(parts) == 4:
                    flog.write(line)
                else:
                    flog.write("\n")
                    flog.flush()
                    capture_atoms = False
                    step_counter += 1

        process.wait()
        print(f"Process exited with code: {process.returncode}")

    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)
    log_msg(f"{os.path.basename(exe)} finished in {m}m {s}s")

# ================== JOB HANDLER ==================
def prepare_job(path):
    import os, textwrap
    # ========== Adjustable simulation parameters (not embedded yet) ==========
    # Electron SCF control
    conv_thr = 1.0e-6             # SCF convergence threshold
    mixing_beta = 0.10            # Density mixing factor
    mixing_mode = "local-TF"      # Mixing scheme
    mixing_ndim = 4              # History length
    electron_maxstep = 200        # Max SCF steps
    diagonalization = "david"     # SCF diagonalization method

    # Smearing / occupations control
    occupations = "smearing"      # 'smearing' or 'fixed'
    smearing = "mv"               # Marzari–Vanderbilt smearing
    degauss = 0.03                # Smearing width (Ry)

    # Geometry relaxation thresholds
    etot_conv_thr = '1.0d-5'
    forc_conv_thr = '1.0d-3'
    ion_dynamics = "bfgs"

    # Optional ESM and magnetization
    assume_isolated = None        # e.g., 'esm' or None
    nspin = 1                     # 1=non-magnetic, 2=spin-polarized
    starting_magnetization = {}   # dict like {'Fe': 0.5, 'O': 0.1}

    # ========================================================================

    prefix, qlabel, atoms, species = "job", "Medium", [], set()
    sys_type, cell_lines, kpoints_block, calc_type = "molecule", [], None, "scf"

    # New parameters (stored but not embedded)
    cell_factor = None
    xc_family, pseudo_type = None, None

    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                key, val = [x.strip().lower() for x in line[1:].split("=", 1)]
                if key == "prefix": prefix = val
                elif key == "quality": qlabel = val.capitalize()
                elif key == "type": sys_type = val
                elif key == "calculation": calc_type = val
                elif key == "cell_factor": cell_factor = float(val)
                elif key == "xc_family": xc_family = val.upper()
                elif key == "pseudo_type": pseudo_type = val.upper()
            elif line.strip().startswith("CELL_PARAMETERS"):
                cell_lines.append(line.strip())
                for _ in range(3):
                    cell_lines.append(next(f).strip())
            elif line.strip().startswith("K_POINTS"):
                kpoints_block = line.strip() + "\n" + next(f).strip()
            elif line.strip():
                el, x, y, z = line.split()
                atoms.append((el, float(x), float(y), float(z)))
                species.add(el)

    # --- Default XC family & type if not specified ---
    if not xc_family: xc_family = "PBE"
    if not pseudo_type: pseudo_type = "USPP"

    # --- Norm-based cutoffs ---
    ecutwfc, ecutrho, kpoints_default = get_cutoffs(xc_family, pseudo_type, qlabel)

    # --- KPOINTS ---
    if kpoints_default.lower() == "gamma":
        kpoints_text = "K_POINTS gamma"
    else:
        kpoints_text = f"K_POINTS automatic\n{kpoints_default}"

    # --- Output next to input file ---
    out_root = os.path.dirname(os.path.abspath(path))
    out_root = out_root.replace("\\", "/")
    os.makedirs(out_root, exist_ok=True)

    # Default filenames
    base_in = os.path.join(out_root, prefix)
    scf_in = base_in + ".in"
    scf_out = base_in + ".out"

    # === (1) Detect all relax logs: prefix_relax_steps.log, prefix_1_relax_steps.log, ... ===
    relax_logs = []
    for f in os.listdir(out_root):
        if f.startswith(prefix) and f.endswith("_relax_steps.log"):
            relax_logs.append(os.path.join(out_root, f))

    # Helper: extract numeric suffix from file name
    def extract_index(path):
        name = os.path.basename(path)
        # prefix_relax_steps.log
        if name == f"{prefix}_relax_steps.log":
            return 0
        # prefix_3_relax_steps.log
        try:
            mid = name[len(prefix)+1:].split("_")[0]
            return int(mid)
        except:
            return 0

    # Sort logs by suffix (ascending)
    relax_logs.sort(key=extract_index)

    # === (2) Backtracking: find latest relax_log with valid geometry ===
    last_positions = None
    chosen_log = None

    for log_path in reversed(relax_logs):  # iterate from newest → oldest
        if not os.path.getsize(log_path) > 0:
            continue

        with open(log_path) as f:
            lines = f.readlines()

        # Try extracting last ATOMIC_POSITIONS block
        block = []
        inside = False
        for line in lines:
            if line.strip().lower().startswith("atomic_positions"):
                inside = True
                block = []
                continue
            if inside:
                if not line.strip():
                    inside = False
                else:
                    block.append(line.strip())

        # If block is valid → accept this log
        if block:
            last_positions = block
            chosen_log = log_path
            break  # stop backtracking

    # Flag for resume mode
    relax_step_continue = last_positions is not None

    # === (3) Override atoms using recovered last_positions ===
    if relax_step_continue:
        atoms = []
        for line in last_positions:
            parts = line.split()
            if len(parts) == 4:
                el = parts[0]
                x, y, z = map(float, parts[1:])
                atoms.append((el, x, y, z))
        # species unchanged
    else:
        chosen_log = None  # no usable resume geometry


    # === (4) Find next available input name: prefix_1.in, prefix_2.in, ... ===
    counter = 1
    while os.path.exists(f"{base_in}_{counter}.in"):
        counter += 1

    scf_in = f"{base_in}_{counter}.in"
    scf_out = f"{base_in}_{counter}.out"

    species_str = "\n".join([
        f" {s} 1.0 {os.path.basename(pseudopotentials[s]['file'])}"
        if s in pseudopotentials else f" {s} 1.0 {s}.upf"
        for s in species
    ])

    pos_str = "\n".join([f" {a[0]} {a[1]} {a[2]} {a[3]}" for a in atoms])
    cell = "\n".join(cell_lines) if cell_lines else (
        "CELL_PARAMETERS angstrom\n 20 0 0\n 0 20 0\n 0 0 20" if sys_type == "molecule" else ""
    )

    # Ion block
    if calc_type in ("relax", "vc-relax") or calc_type.startswith("vc-relax"):
        ions_block = textwrap.dedent(f"""
        &ions
          ion_dynamics = '{ion_dynamics}'
        /
        """)
    else:
        ions_block = ""

    #Electron block
    electrons_block = textwrap.dedent(f"""
    &electrons
      conv_thr = {conv_thr},
      mixing_beta = {mixing_beta},
      mixing_mode = '{mixing_mode}',
      mixing_ndim = {mixing_ndim},
      electron_maxstep = {electron_maxstep},
      diagonalization = '{diagonalization}',
    /
    """)

    # --- Directional vc-relax handling ---
    if calc_type.startswith("vc-relax"):
        axis_part = None
        if ":" in calc_type:
            calc_main, axis_part = [c.strip() for c in calc_type.split(":", 1)]
        else:
            calc_main = calc_type

        cell_dofree = None
        if axis_part:
            axes = [a.strip().lower() for a in axis_part.split(",")]
            valid_axes = {"x", "y", "z", "xy", "xz", "yz", "xyz"}
            if len(axes) == 1 and axes[0] in {"x", "y", "z"}:
                cell_dofree = axes[0]
            elif set(axes) <= {"x", "y"}:
                cell_dofree = "xy"
            elif set(axes) <= {"x", "z"}:
                cell_dofree = "xz"
            elif set(axes) <= {"y", "z"}:
                cell_dofree = "yz"
            elif set(axes) == {"x", "y", "z"}:
                cell_dofree = "xyz"

        if cell_factor:
            cell_block = textwrap.dedent(f"""
            &cell
            cell_dynamics = 'bfgs',
            cell_factor = {cell_factor:.3f}{"," if cell_dofree else ""}
            {"cell_dofree = '" + cell_dofree + "'" if cell_dofree else ""}
            /
            """)
        else:
            cell_block = textwrap.dedent(f"""
            &cell
            cell_dynamics = 'bfgs'{"," if cell_dofree else ""}
            {"cell_dofree = '" + cell_dofree + "'" if cell_dofree else ""}
            /
            """)
    else:
        cell_block = ""
        calc_main = calc_type  # ensure defined

    # ---- Correct QE namelist formatting ----
    system_lines = [
        f"ibrav = 0",
        f"nat = {len(atoms)}",
        f"ntyp = {len(species)}",
        f"ecutwfc = {ecutwfc}",
        f"ecutrho = {ecutrho}",
        "input_dft = 'PBE'",
        f"occupations = '{occupations}'",
        f"smearing = '{smearing}'",
        f"degauss = {degauss}"
    ]

    system_block = "  " + ",\n  ".join(system_lines)
    pseudo_dir_qe = pseudo_dir_selected.replace("\\", "/")
    species_block = f"ATOMIC_SPECIES\n{species_str}"
    pos_block = f"ATOMIC_POSITIONS angstrom\n{pos_str}"
    restart_mode = 'from_scratch'
    scf_text = textwrap.dedent(f"""
    &control
      calculation = '{calc_main}',
      restart_mode = '{restart_mode}',
      prefix = '{prefix}_{counter}',
      pseudo_dir = '{pseudo_dir_qe}',
      outdir = '{out_root}',
      etot_conv_thr = {etot_conv_thr},
      forc_conv_thr = {forc_conv_thr}
    /
    &system
    {system_block}
    /
    {electrons_block}
    {ions_block}{cell_block}
    {cell}
    {species_block}
    {pos_block}
    {kpoints_text}
    """)
    with open(scf_in, "w") as f:
        f.write(scf_text)

    # Label for scheduler
    if relax_step_continue:
        restart_mode = f"resume_from_log_{extract_index(chosen_log)}"
    else:
        restart_mode = "from_scratch"
    return {
        "prefix": f"{prefix}_{counter}",
        "scf_in": scf_in,
        "scf_out": scf_out,
        "mode": restart_mode,
        "params": {
            "conv_thr": conv_thr,
            "mixing_beta": mixing_beta,
            "mixing_mode": mixing_mode,
            "mixing_ndim": mixing_ndim,
            "electron_maxstep": electron_maxstep,
            "diagonalization": diagonalization,
            "occupations": occupations,
            "smearing": smearing,
            "degauss": degauss,
            "etot_conv_thr": etot_conv_thr,
            "forc_conv_thr": forc_conv_thr,
            "ion_dynamics": ion_dynamics,
            "assume_isolated": assume_isolated,
            "nspin": nspin,
            "starting_magnetization": starting_magnetization
        }
    }

# ----------------- Helpers -----------------

JOB_DONE_RE = re.compile(r"\bjob\s*done\b", re.IGNORECASE)

def any_out_has_job_done(out_files: list[str]) -> bool:
    """Return True if any .out file contains 'JOB DONE' (case-insensitive)."""
    for out_path in out_files:
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if JOB_DONE_RE.search(line):
                        return True
        except Exception:
            pass
    return False

def is_empty_file(filepath: str) -> bool:
    """
    Returns True if file is missing OR empty (0 bytes or only whitespace).
    This matches your current behavior (missing treated as empty).
    """
    if not os.path.isfile(filepath):
        return True
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return len(f.read().strip()) == 0
    except Exception:
        return True


def cpu_time_exceeded_in_any_out(out_files, threshold_s=360000.0) -> tuple[bool, float | None, str | None]:
    """
    Scan each .out file for the line:
      'total cpu time spent up to now is'
    Return (exceeded, secs, filename) for the first exceeding match found.
    """
    needle = "total cpu time spent up to now is"
    for out_path in out_files:
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if needle in line.lower():
                        # Your original logic: float(line.strip().split()[-2])
                        parts = line.strip().split()
                        secs = float(parts[-2])
                        if secs > threshold_s:
                            return True, secs, os.path.basename(out_path)
                        break
        except Exception:
            pass
    return False, None, None


def collect_subfolder_inputs(compound_dir: str) -> list[str]:
    """
    For each immediate subfolder of compound_dir, collect all *.txt inside that subfolder.
    (If your .txt are deeper, adjust this to walk recursively.)
    """
    inputs = []
    for name in os.listdir(compound_dir):
        sub = os.path.join(compound_dir, name)
        if not os.path.isdir(sub):
            continue
        for f in os.listdir(sub):
            if f.endswith(".txt"):
                inputs.append(os.path.join(sub, f))
    return inputs


def select_subfolders(compound_dir: str) -> list[str]:
    """
    Sort subfolders.
    """
    subs = sorted(
        os.path.join(compound_dir, d)
        for d in os.listdir(compound_dir)
        if os.path.isdir(os.path.join(compound_dir, d))
    )
    return subs


# ----------------- Scheduler -----------------

def schedule_jobs_simple(input_files: list[str], max_cores: int = 32) -> None:
    """
    Simplified scheduler:
      - ncores is always 1
      - max_cores default = 32
      - keeps CPU time skip check
    """
    ncores = 1
    available = max_cores
    lock = threading.Lock()

    def run_job(job: dict, job_cores: int = 1) -> None:
        nonlocal available

        prefix = job.get("prefix", "job")
        mode = job.get("mode", "unknown")
        scf_in, scf_out = job["scf_in"], job["scf_out"]

        log_msg(f"[START] Job {prefix} ({mode}) ncores={job_cores}, available={available}")
        run_qe_with_timer(pw_exe, scf_in, scf_out, nprocs=job_cores)

        with lock:
            available += job_cores

        log_msg(f"[END] Job {prefix} finished, released {job_cores} cores (available={available})")


    pending = input_files.copy()
    futures = {}

    log_msg(f"[INFO] Scheduler started with {max_cores} total cores (ncores fixed to 1).")

    with ThreadPoolExecutor() as exe:
        while pending or futures:
            with lock:
                # Try queue as many as possible
                for path in pending[:]:
                    if available < ncores:
                        break  # no cores available right now

                    # -------- CPU TIME SKIP CHECK (multiple out files) --------
                    job = prepare_job(path)
                    prefix = job.get("prefix", "job")

                    work_dir = os.path.dirname(path)
                    save_path = os.path.join(work_dir, f"{prefix}.save")
                    relax_log = os.path.join(work_dir, f"{prefix}_relax_steps.log")

                    # Collect all matching out files for the same base name as the .txt
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_files = [
                        os.path.join(work_dir, f)
                        for f in os.listdir(work_dir)
                        if f.startswith(base) and f.endswith(".out")
                    ]

                    # --------------- FIRST CHECK: JOB DONE in any .out ---------------
                    if out_files and any_out_has_job_done(out_files):
                        log_msg(f"[SKIP] {os.path.basename(path)}: found 'JOB DONE' in an .out → Skipping job.")
                        pending.remove(path)
                        continue

                    # --------------- SECOND CHECK: CPU time exceeded (your existing logic) ---------------
                    relax_log_empty = is_empty_file(relax_log)

                    if (not os.path.isdir(save_path)) and out_files and relax_log_empty:
                        exceeded, secs, which = cpu_time_exceeded_in_any_out(out_files, threshold_s=360000.0)
                        if exceeded:
                            log_msg(
                                f"[SKIP] {os.path.basename(path)} found CPU={secs:.1f}s "
                                f"in {which} (>360000s) → Skipping job."
                            )
                            pending.remove(path)
                            continue

                    # Queue job (ncores fixed to 1)
                    available -= ncores
                    fut = exe.submit(run_job, job, ncores)
                    futures[fut] = (path, ncores)
                    pending.remove(path)
                    log_msg(f"[QUEUE] Allocated {os.path.basename(path)}, available={available}")

            # Reclaim completed
            if futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    path, job_cores = futures.pop(fut)
                    # Note: cores also returned in run_job; keeping this simple:
                    # do NOT add again here to avoid double-release.
                    log_msg(f"[DONE] Completed {os.path.basename(path)}")
            else:
                time.sleep(0.25)

    log_msg("[INFO] All jobs finished.")


# ----------------- Main -----------------

if __name__ == "__main__":
    base_dir = os.getcwd()
    compound_dir = os.path.join(base_dir, "compound")

    if not os.path.isdir(compound_dir):
        raise FileNotFoundError(f"Expected folder not found: {compound_dir}")

    # Even-indexed subfolders for this process; run another script with odd_index=True
    subfolders = select_subfolders(compound_dir)

    all_inputs = []
    for sub in subfolders:
        txts = [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".txt")]
        txts.sort()
        all_inputs.extend(txts)

    all_inputs.sort()

    if not all_inputs:
        log_msg("[INFO] No pending .txt jobs remaining.")
        raise SystemExit(0)

    log_msg(f"[INFO] Total jobs to schedule (even-indexed subfolders): {len(all_inputs)}")
    schedule_jobs_simple(all_inputs, max_cores=32)


