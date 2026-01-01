import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import time
from multiprocessing import Pool, cpu_count

# Import Shared Core
from ca_core import CAEngine, CycleFinder, add_label

OUTPUT_DIR = "comparison_output"
TIME_LIMIT = 60
MAX_N = 500

def run_algo_wrapper(func, x0, width, bank):
    if bank <= 0: return False, 0
    t0 = time.perf_counter()
    try:
        # Pass bank as timeout so the engine kills it internally if it gets stuck
        func(x0, width, timeout=bank)
        dt = time.perf_counter() - t0
        return True, dt
    except:
        return False, 0

def process_comparison(rule):
    engine = CAEngine(rule)
    finder = CycleFinder(engine)
    
    runners = {
        'Brent':   {'func': finder.brent,   'data': [], 'active': True, 'bank': TIME_LIMIT},
        'Floyd':   {'func': finder.floyd,   'data': [], 'active': True, 'bank': TIME_LIMIT},
        'Hashmap': {'func': finder.hashmap, 'data': [], 'active': True, 'bank': TIME_LIMIT},
        'Gosper':  {'func': finder.gosper,  'data': [], 'active': True, 'bank': TIME_LIMIT}
    }
    
    n = 3
    while n < MAX_N:
        all_dead = True
        start_state = 1 << (n // 2)
        
        for name, r in runners.items():
            if r['active']:
                all_dead = False
                success, dt = run_algo_wrapper(r['func'], start_state, n, r['bank'])
                
                if success:
                    r['bank'] -= dt
                    r['data'].append((n, dt))
                else:
                    r['active'] = False
        
        if all_dead: break
        n += 2
        
    return rule, runners

def plot_grid(results):
    fig, axes = plt.subplots(16, 16, figsize=(24, 24))
    
    colors = {
        'Brent': '#1f77b4',   # Blue
        'Floyd': '#d62728',   # Red
        'Hashmap': '#2ca02c', # Green
        'Gosper': '#9467bd'   # Purple
    }

    for rule in range(256):
        ax = axes.flat[rule]
        res = results.get(rule, {})
        has_data = False
        
        for algo, info in res.items():
            if info['data']:
                has_data = True
                ns, ts = zip(*info['data'])
                ax.plot(ns, ts, color=colors[algo], ls='-', lw=1.2)
        
        ax.set_yscale('log')
        ax.set_facecolor('#fafafa')
        
        ax.axis('off')
        
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                             linewidth=0.5, edgecolor='#666666', facecolor='#fafafa', zorder=-1)
        ax.add_patch(rect)
        
        add_label(ax, str(rule), 0.05, 0.95, 8, 'left', 'top')
        
        if not has_data:
            add_label(ax, "Timeout", 0.5, 0.5, 6, 'center', 'center')

    # Legend
    handles = [plt.Line2D([],[], color=colors[k], ls='-', lw=2, label=k) for k in colors]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.96))
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.1, hspace=0.1)
    fig.savefig(os.path.join(OUTPUT_DIR, "algo_comparison.png"), dpi=200)
    plt.close(fig)

if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting Algorithm Comparison on {cpu_count()} cores...")
    results = {}
    with Pool(cpu_count()) as pool:
        for i, (r, d) in enumerate(pool.imap_unordered(process_comparison, range(256))):
            results[r] = d
            print(f"  > {i}/256 done...", end='\r')
        print(f"  > 256/256 done...", end='\r')
        
            
    print("\nGenerating Grid...")
    plot_grid(results)
    print("Done.")