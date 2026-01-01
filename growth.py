import matplotlib
matplotlib.use('Agg') 

import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects
from multiprocessing import Pool, cpu_count
from matplotlib import colors
import os
import itertools
import shutil
import math
from collections import defaultdict
from ca_core import *

# CONSTANTS & CONFIG
OUTPUT_DIR = "growth_output"
TIME_LIMIT = 180
MAX_N = 500

CYCLES_DIR = os.path.join(OUTPUT_DIR, "cycles_graphs")
TRANSIENTS_DIR = os.path.join(OUTPUT_DIR, "transients_graphs")
META_DIR = os.path.join(OUTPUT_DIR, "meta_visuals")

# CORE PROCESSING
def process_rule(args):
    rule, limit, cap = args
    engine = CAEngine(rule)
    finder = CycleFinder(engine)
    data = {'n': [], 'mu': [], 'lam': [], 'max_n': 0}
    
    n = 5
    while n <= cap:
        start = 1 << (n // 2)
        
        mu, lam = finder.brent(start, n, timeout=limit)
        
        # If None, the internal timeout triggered
        if mu is None: 
            break

        data['n'].append(n)
        data['mu'].append(mu)
        data['lam'].append(lam)
        data['max_n'] = n
        n += 2
    return rule, data

def run_experiment(limit=0.5, cap=100):
    tasks = [(r, limit, cap) for r in range(256)]
    results = {}
    total = len(tasks)
    print(f"Starting pool with {cpu_count()} cores for {total} rules...")
    
    with Pool(cpu_count()) as pool:
        for i, (r, d) in enumerate(pool.imap_unordered(process_rule, tasks)):
            results[r] = d
            print(f"  > Progress: {i}/{total} rules processed...", end='\r')
    print(f"  > Progress: {total}/{total} rules processed.          ")
    return results

# PLOTTING HELPERS
def add_labels(ax, rule, tup_txt="", cls_txt="", top_char=""):
    path_fx = [PathEffects.withStroke(linewidth=STROKE_W, foreground='white')]
    
    t = ax.text(0.05, 0.95, str(rule), transform=ax.transAxes, fontsize=FONT_L, 
                fontweight='bold', ha='left', va='top')
    t.set_path_effects(path_fx)

    if top_char:
        t = ax.text(0.5, 0.95, top_char, transform=ax.transAxes, fontsize=FONT_L, 
                    fontweight='bold', ha='center', va='top')
        t.set_path_effects(path_fx)

    if tup_txt:
        t = ax.text(0.5, 0.05, tup_txt, transform=ax.transAxes, fontsize=FONT_M, 
                    fontweight='bold', ha='center', va='bottom')
        t.set_path_effects(path_fx)

    if cls_txt:
        t = ax.text(0.95, 0.05, cls_txt, transform=ax.transAxes, fontsize=FONT_S, 
                    fontweight='bold', ha='right', va='bottom')
        t.set_path_effects(path_fx)

def clean_plot(ax, bg_color='white'):
    ax.axis('off')
    # Draw manual border box matching the axis extent
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                         linewidth=0.5, edgecolor='#666666', facecolor=bg_color, zorder=-1)
    ax.add_patch(rect)

def plot_smart(ax, rule, data, key, is_cyc, is_grid=False):
    cls = -1
    bg = 'white'
    
    if data and len(data['n']) > 0:
        cls, bg = classify(data['n'], data[key])
        
        ax.plot(data['n'], data[key], marker='o', markersize=2, color='#222222', lw=1)
        ax.set_yscale('log')
        
        mx_v = data[key][-1] if data[key] else 0
        tup = f"({data['max_n']}, {format_num(mx_v)})"
        
        char = ""
        if not is_grid: char = "L" if is_cyc else "T"
            
        add_labels(ax, rule, tup, SHORT_NAMES[cls], char)
    else:
        # Timeout / Empty
        add_labels(ax, rule, cls_txt="-")

    clean_plot(ax, bg)
    return cls

def plot_meta(ax, rule, data, cmap):
    t_cls, c_cls = -1, -1
    bg = '#ffffff'
    cls_str = "-"
    
    if data:
        t_cls, _ = classify(data['n'], data['mu'])
        c_cls, _ = classify(data['n'], data['lam'])
        
        palette = [
            ['#ffffff', '#e6f2ff', '#fffacd', '#ffe6e6'], 
            ['#e6ffe6', '#ccebff', '#fff5b3', '#ffcccc'], 
            ['#ccffcc', '#b3e0ff', '#ffeebb', '#ffb3b3'], 
            ['#99ff99', '#99d6ff', '#ffe499', '#ff9999'] 
        ]
        try: bg = palette[t_cls][c_cls]
        except: pass
        cls_str = f"{SHORT_NAMES[t_cls]}-{SHORT_NAMES[c_cls]}"

    width = 51
    state = 1 << (width // 2)
    hist = []
    eng = CAEngine(rule)
    for _ in range(50):
        hist.append([(state >> i) & 1 for i in range(width)][::-1])
        state = eng.evolve(state, width)
    
    # Render the CA visual
    ax.imshow(np.array(hist), cmap=cmap, aspect='equal', interpolation='nearest', zorder=1)
    add_labels(ax, rule, cls_txt=cls_str)
    
    clean_plot(ax, bg)
    return t_cls, c_cls

# AGGREGATION & SAVING
def generate_class_summary(class_dict, res_data, base_dir, mode_type, cmap=None):
    cols = 4
    
    for class_name, rules in class_dict.items():
        if not rules: continue
        
        n_plots = len(rules)
        rows = math.ceil(n_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
        axes_flat = axes.flatten()
        
        print(f"  > Creating summary for '{class_name}' ({n_plots} graphs)...")
        
        for idx, ax in enumerate(axes_flat):
            if idx < n_plots:
                rule_idx = rules[idx]
                d = res_data.get(rule_idx)
                
                if mode_type == 'cycles':
                    plot_smart(ax, rule_idx, d, 'lam', True, is_grid=False)
                elif mode_type == 'transients':
                    plot_smart(ax, rule_idx, d, 'mu', False, is_grid=False)
                elif mode_type == 'meta':
                    plot_meta(ax, rule_idx, d, cmap)
            else:
                ax.axis('off') # Hide empty slots
        
        plt.subplots_adjust(wspace=0.1, hspace=0.15)
        
        filename = f"Combined_{class_name}.png"
        save_path = os.path.join(base_dir, class_name, filename)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

def save_grids(res):
    # 1. Cycles & Transients
    for is_cyc in [True, False]:
        key = 'lam' if is_cyc else 'mu'
        name = "Cycles" if is_cyc else "Transients"
        folder = CYCLES_DIR if is_cyc else TRANSIENTS_DIR
        
        class_tracker = defaultdict(list)
        
        print(f"Generating {name} Grid...")
        fig, axes = plt.subplots(16, 16, figsize=(24, 24))
        
        for i, ax in enumerate(axes.flat):
            d = res.get(i)
            # Main Grid Plot
            plot_smart(ax, i, d, key, is_cyc, True)
            
            # Individual Plot
            if d and d['n']:
                f_s, a_s = plt.subplots(figsize=(4,4))
                c = plot_smart(a_s, i, d, key, is_cyc, False)
                a_s.set_position([0,0,1,1]) # Full bleed
                
                sub = LONG_NAMES[c] if 0<=c<=3 else "Unknown"
                class_tracker[sub].append(i)
                
                f_s.savefig(f"{folder}/{sub}/Rule_{i:03d}_{name}.png", dpi=300)
                plt.close(f_s)
            else:
                class_tracker["Unknown"].append(i)
                
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.1, hspace=0.1)
        fig.suptitle(f"{name} Growth Analysis", fontsize=24)
        fig.savefig(f"{OUTPUT_DIR}/{name.lower()}_grid.png", dpi=300)
        plt.close(fig)
        
        # Combined Summaries
        mode = 'cycles' if is_cyc else 'transients'
        generate_class_summary(class_tracker, res, folder, mode)

    # 2. Meta Visuals
    print("Generating Meta Visuals...")
    fig, axes = plt.subplots(16, 16, figsize=(24, 24))
    cmap = colors.ListedColormap(['#00000000', '#000000'])
    
    meta_tracker = defaultdict(list)
    
    for i, ax in enumerate(axes.flat):
        d = res.get(i)
        t, c = plot_meta(ax, i, d, cmap)
        
        if d:
            f_s, a_s = plt.subplots(figsize=(4,4))
            plot_meta(a_s, i, d, cmap)
            a_s.set_position([0,0,1,1])
            
            if 0<=t<=3 and 0<=c<=3: 
                sub = f"{LONG_NAMES[t]}-{LONG_NAMES[c]}"
            else: 
                sub = "Unknown"
            
            meta_tracker[sub].append(i)
            
            f_s.savefig(f"{META_DIR}/{sub}/Rule_{i:03d}_Visual.png", dpi=300)
            plt.close(f_s)
        else:
            meta_tracker["Unknown"].append(i)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.05)
    fig.suptitle("Rule Complexity Visuals", fontsize=24)
    fig.savefig(f"{OUTPUT_DIR}/meta_visual_grid.png", dpi=300)
    plt.close(fig)
    
    # Combined Meta Summaries
    generate_class_summary(meta_tracker, res, META_DIR, 'meta', cmap)

# MAIN EXECUTION
if __name__ == '__main__':
    # 1. SETUP DIRECTORIES
    if os.path.exists(OUTPUT_DIR):
        try: shutil.rmtree(OUTPUT_DIR)
        except OSError as e: print(f"Error removing directory: {e}")

    for d in [OUTPUT_DIR, CYCLES_DIR, TRANSIENTS_DIR, META_DIR]:
        os.makedirs(d, exist_ok=True)

    for name in LONG_NAMES:
        os.makedirs(os.path.join(CYCLES_DIR, name), exist_ok=True)
        os.makedirs(os.path.join(TRANSIENTS_DIR, name), exist_ok=True)

    for t_name, c_name in itertools.product(LONG_NAMES, LONG_NAMES):
        os.makedirs(os.path.join(META_DIR, f"{t_name}-{c_name}"), exist_ok=True)

    for d in [CYCLES_DIR, TRANSIENTS_DIR, META_DIR]:
        os.makedirs(os.path.join(d, "Unknown"), exist_ok=True)

    print(f"Directories created in '{OUTPUT_DIR}/'")

    # 2. RUN ANALYSIS
    print("--- Starting Analysis ---")
    # Setting limit to 180s (3 minutes) per rule to ensure completion
    # Setting cap to 1000 cells max width
    data = run_experiment(limit=TIME_LIMIT, cap=MAX_N)
    save_grids(data)
    print("--- Done ---")