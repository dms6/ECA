import time
import numpy as np
import matplotlib.patheffects as PathEffects

# CONSTANTS & CONFIG
LONG_NAMES = ['Constant', 'Linear', 'Quadratic', 'Exponential']
SHORT_NAMES = ['C', 'L', 'Q', 'E']
FONT_L, FONT_M, FONT_S = 10, 8, 9
STROKE_W = 2.5

# ENGINE
class CAEngine:
    def __init__(self, rule_number):
        self.rule_bits = [(rule_number >> k) & 1 for k in range(8)]

    def evolve(self, state, width):
        mask = (1 << width) - 1
        # Wraparound logic
        L = ((state >> 1) | (state << (width - 1))) & mask
        R = ((state << 1) | (state >> (width - 1))) & mask
        C = state
        
        n = 0
        if self.rule_bits[0]: n |= (~L & ~C & ~R)
        if self.rule_bits[1]: n |= (~L & ~C &  R)
        if self.rule_bits[2]: n |= (~L &  C & ~R)
        if self.rule_bits[3]: n |= (~L &  C &  R)
        if self.rule_bits[4]: n |= ( L & ~C & ~R)
        if self.rule_bits[5]: n |= ( L & ~C &  R)
        if self.rule_bits[6]: n |= ( L &  C & ~R)
        if self.rule_bits[7]: n |= ( L &  C &  R)
        return n & mask

# CYCLE ALGORITHMS
class CycleFinder:
    def __init__(self, engine):
        self.engine = engine

    def _check_timeout(self, start_time, limit, step_cnt):
        # Check every 4096 steps to minimize overhead
        if limit and (step_cnt & 4095 == 0):
            if time.perf_counter() - start_time > limit:
                return True
        return False

    def brent(self, x0, width, timeout=None):
        t0 = time.perf_counter()
        power = 1
        lam = 1
        tortoise = x0
        hare = self.engine.evolve(x0, width)
        steps = 0

        # 1. Find Lambda
        while tortoise != hare:
            if self._check_timeout(t0, timeout, steps): return None, None
            if power == lam:
                tortoise = hare
                power *= 2
                lam = 0
            hare = self.engine.evolve(hare, width)
            lam += 1
            steps += 1
            
        tortoise = hare = x0
        
        # 2. Reset Hare
        for i in range(lam):
            if self._check_timeout(t0, timeout, i): return None, None
            hare = self.engine.evolve(hare, width)
            
        # 3. Find Mu
        mu = 0
        while tortoise != hare:
            if self._check_timeout(t0, timeout, mu): return None, None
            tortoise = self.engine.evolve(tortoise, width)
            hare = self.engine.evolve(hare, width)
            mu += 1
            
        return mu, lam

    def floyd(self, x0, width, timeout=None):
        t0 = time.perf_counter()
        steps = 0
        tortoise = self.engine.evolve(x0, width)
        hare = self.engine.evolve(self.engine.evolve(x0, width), width)
        
        while tortoise != hare:
            if self._check_timeout(t0, timeout, steps): return None, None
            tortoise = self.engine.evolve(tortoise, width)
            hare = self.engine.evolve(self.engine.evolve(hare, width), width)
            steps += 1
        
        mu = 0
        tortoise = x0
        while tortoise != hare:
            if self._check_timeout(t0, timeout, mu): return None, None
            tortoise = self.engine.evolve(tortoise, width)
            hare = self.engine.evolve(hare, width)
            mu += 1
            
        lam = 1
        hare = self.engine.evolve(tortoise, width)
        while tortoise != hare:
            if self._check_timeout(t0, timeout, lam): return None, None
            hare = self.engine.evolve(hare, width)
            lam += 1
        return mu, lam

    def hashmap(self, x0, width, timeout=None):
        t0 = time.perf_counter()
        seen = {x0: 0}
        curr = x0
        step = 0
        while True:
            if self._check_timeout(t0, timeout, step): return None, None
            curr = self.engine.evolve(curr, width)
            step += 1
            if curr in seen:
                mu = seen[curr]
                lam = step - mu
                return mu, lam
            seen[curr] = step

    def gosper(self, x0, width, timeout=None):
        t0 = time.perf_counter()
        saved = {0: x0}
        k = 1
        curr = x0
        n = 0
        while True:
            if self._check_timeout(t0, timeout, n): return None, None
            curr = self.engine.evolve(curr, width)
            n += 1
            
            # Check against saved checkpoints
            for idx, val in saved.items():
                if val == curr:
                    # Detected a loop multiple. 
                    # Reverting to Brent-like cleanup for exactness
                    return self._resolve_exact(x0, width, idx, n - idx, timeout, t0)
            
            if n == k:
                saved[n] = curr
                k *= 2

    def _resolve_exact(self, x0, width, start_est, len_est, timeout, t0):
        # Helper for Gosper to refine the cycle
        hare = x0
        for _ in range(len_est):
            hare = self.engine.evolve(hare, width)
        
        mu = 0
        tortoise = x0
        while tortoise != hare:
            if self._check_timeout(t0, timeout, mu): return None, None
            tortoise = self.engine.evolve(tortoise, width)
            hare = self.engine.evolve(hare, width)
            mu += 1
        
        lam = 1
        hare = self.engine.evolve(tortoise, width)
        while tortoise != hare:
            if self._check_timeout(t0, timeout, lam): return None, None
            hare = self.engine.evolve(hare, width)
            lam += 1
        return mu, lam

def format_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n > 10_000: return f"{n/1_000:.0f}k"
    return str(n)

def classify(x, y):
    if not len(y): return 0, '#ffffff'
    arr_y = np.array(y)
    if len(arr_y) == 0 or np.all(arr_y == 0): return 0, '#e6ffe6'
    if np.median(y) == arr_y[-1]: return 0, '#e6ffe6'
    
    log_x = np.log10(x)
    log_y = np.log10(np.where(arr_y <= 0, 0.1, arr_y))
    if len(log_x) < 2: return 0, '#e6ffe6'

    try: k, _ = np.polyfit(log_x, log_y, 1)
    except: return 0, '#e6ffe6'

    if 1.5 < k < 2.5: return 2, '#fffacd' # Quadratic
    if k >= 2.5: return 3, '#ffe6e6'      # Exponential
    return 1, '#e6f2ff'                   # Linear

def add_label(ax, txt, x, y, size, align='center', va='center'):
    path_fx = [PathEffects.withStroke(linewidth=STROKE_W, foreground='white')]
    t = ax.text(x, y, txt, transform=ax.transAxes, fontsize=size, 
                fontweight='bold', ha=align, va=va)
    t.set_path_effects(path_fx)