/*
 * =================================================================================
 * Schwarzschild Geodesic Simulation (Optimized Version)
 *
 * This single file is structured to be easily split into modules in the future.
 * It incorporates performance and maintainability improvements.
 * =================================================================================
 */

// ============================================================
// File: constants.js
// ============================================================
const M = 1.0;
const Rg = 2.0 * M;
const DEFAULT_RTOL = 1e-12;
const DEFAULT_ATOL = 1e-15;
const DEFAULT_TTOTAL = 2000;
const DEFAULT_NFRAMES = 1200;
const DEFAULT_R_ESCAPE = Rg * 200;
const TAU_CHUNK = 10.0;
const F_FLOOR = 1e-14;
const HORIZON_MARGIN = 1.00001;
const MAX_GEODESIC_POINTS = 200000;
const DEFAULT_MAX_STEP_FACTOR = null;
const ANIMATION_INTERVAL_MS = 30;

// ============================================================
// File: utils.js
// ============================================================
function linspace(a, b, n) {
  if (n <= 1) return [a];
  const out = new Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; ++i) out[i] = a + step * i;
  return out;
}
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }


// ============================================================
// File: physics.js
// (Metric and Geodesic Equations)
// ============================================================
function metricElements(r) {
  const f = 1.0 - Rg / r;
  return {f: f, g_tt: -f, g_rr: 1.0 / f, g_pp: r*r};
}
// y = [r, ur, phi, tcoord] -> returns dy/dtau
function geodesic_tau(tau, y, L, E) {
  const r = y[0], ur = y[1];
  const f = 1.0 - Rg / r;
  const d2r = (Rg / (2.0 * f * r*r)) * (- E*E + ur*ur) + f * (L*L) / (r*r*r);
  const dphi = L / (r*r);
  const dt = E / f;
  return [ur, d2r, dphi, dt];
}

// ============================================================
// File: dopri.js
// (Dormand-Prince RK45 Integrator - Refactored as an object)
// ============================================================
const DOPRI = {
  // Butcher tableau coefficients
  c: [0, 1/5, 3/10, 4/5, 8/9, 1, 1],
  a: [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
  ],
  b5: [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
  b4: [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],

  integrate(f, tau0, y0, tau1, argsObj) {
    const atol = argsObj.atol ?? DEFAULT_ATOL;
    const rtol = argsObj.rtol ?? DEFAULT_RTOL;
    const r_escape = argsObj.r_escape ?? DEFAULT_R_ESCAPE;
    const max_step = argsObj.max_step ?? Infinity;
    const maxPoints = argsObj.maxPoints ?? 1000000;

    const outTau = [];
    const outY = [];
    let tau = tau0;
    let y = [...y0];
    let h = Math.min(1.0, (tau1 - tau0) / 100.0, max_step);
    const min_h = 1e-12;
    const maxIter = 1000000;
    let iter = 0;
    const k = new Array(7);

    while (tau < tau1 - 1e-15 && iter < maxIter && outTau.length < maxPoints) {
      iter++;
      if (tau + h > tau1) h = tau1 - tau;
      
      k[0] = f(tau, y);
      for(let i = 1; i < 7; i++) {
          let y_temp = [...y];
          for(let j = 0; j < i; j++) {
              for(let l=0; l<4; l++) y_temp[l] += this.a[i][j] * k[j][l] * h;
          }
          k[i] = f(tau + this.c[i] * h, y_temp);
      }

      const y5 = [...y], y4 = [...y];
      const err = [0,0,0,0];
      for(let i=0; i<7; i++) {
        for(let l=0; l<4; l++) {
            y5[l] += this.b5[i] * k[i][l] * h;
            y4[l] += this.b4[i] * k[i][l] * h;
        }
      }
      for(let l=0; l<4; l++) err[l] = y5[l] - y4[l];

      let errNorm = 0;
      for (let i=0;i<4;i++) {
        const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(y5[i]));
        errNorm = Math.max(errNorm, Math.abs(err[i]) / sc);
      }

      if (errNorm <= 1.0) {
        tau += h;
        y = y5;
        outTau.push(tau);
        outY.push([...y]);
        
        const safety = 0.9, exp = 1.0/5.0;
        const factor = Math.max(0.2, Math.min(5.0, safety * Math.pow(1 / Math.max(errNorm, 1e-16), exp)));
        h = Math.min(h * factor, max_step);
        if (h < min_h) h = min_h;
        
        if (y[0] <= HORIZON_MARGIN * Rg) return {tau: outTau, yout: outY, event: "horizon"};
        if (y[0] >= r_escape) return {tau: outTau, yout: outY, event: "escape"};
      } else {
        const safety = 0.9, exp = 1.0/5.0;
        const factor = Math.max(0.1, safety * Math.pow(1/errNorm, exp));
        h *= factor;
        if (h < min_h) h = min_h;
      }
    }
    return {tau: outTau, yout: outY, event: null};
  }
};

// ============================================================
// File: particle.js
// ============================================================
class Particle {
    // Public properties (read-only from outside)
    name; color; x0; y0; vx0; vy0; t_inject; L = 0; E = 0; state = "active"; alive = true;
    x_obs = null; y_obs = null;

    // Private fields
    #rtol; #atol; #r_escape; #max_step_factor;
    #y_init; #u0;
    #tau_vals = []; #r_tau = []; #ur_tau = []; #phi_tau = []; #t_tau = [];
    #t_to_tau_monotone = false;

    constructor(x0, y0, vx0, vy0, t_inject = 0.0, opts = {}) {
        this.x0 = Number(x0); this.y0 = Number(y0);
        this.vx0 = Number(vx0); this.vy0 = Number(vy0);
        this.t_inject = Number(t_inject);
        this.name = opts.name ?? "Noname";
        this.color = opts.color ?? "steelblue";

        this.#rtol = opts.rtol ?? DEFAULT_RTOL;
        this.#atol = opts.atol ?? DEFAULT_ATOL;
        this.#r_escape = opts.r_escape ?? DEFAULT_R_ESCAPE;
        this.#max_step_factor = opts.max_step_factor ?? DEFAULT_MAX_STEP_FACTOR;

        this.#compute_initial_conditions();
        this.#extend_geodesic(TAU_CHUNK);
    }

    #compute_initial_conditions() {
        let { x0, y0, vx0, vy0 } = this;
        let r0 = Math.hypot(x0, y0);
        let phi0 = Math.atan2(y0, x0);

        if (r0 <= HORIZON_MARGIN * Rg) {
            r0 = HORIZON_MARGIN * Rg * 1.0001;
            this.x0 = r0 * Math.cos(phi0); this.y0 = r0 * Math.sin(phi0);
            console.warn(`[warning] r0 too small; bumped to ${r0}`);
        }

        const {f, g_tt, g_rr, g_pp} = metricElements(r0);
        const v_r = Math.cos(phi0)*vx0 + Math.sin(phi0)*vy0;
        const v_tang = -Math.sin(phi0)*vx0 + Math.cos(phi0)*vy0;
        const dphi_dt = v_tang / r0;

        const denom = g_tt + g_rr * (v_r*v_r) + g_pp * (dphi_dt*dphi_dt);
        if (denom >= -1e-12) {
             throw new Error(`[${this.name}] Invalid initial velocity: cannot form timelike 4-velocity.`);
        }
        
        const u_t = Math.sqrt(-1.0 / denom);
        const u_r = u_t * v_r;
        const u_phi = u_t * dphi_dt;
        this.E = -g_tt * u_t;
        this.L = g_pp * u_phi;

        this.#y_init = [r0, u_r, phi0, 0.0];
        this.#u0 = [u_t, u_r, u_phi];
    }

    #extend_geodesic(tau_max = TAU_CHUNK) {
        if (!this.alive) return;
        
        const y0 = this.#tau_vals.length > 0
            ? [this.#r_tau.at(-1), this.#ur_tau.at(-1), this.#phi_tau.at(-1), this.#t_tau.at(-1)]
            : [...this.#y_init];
        const tauStart = this.#tau_vals.at(-1) ?? 0.0;
        const tauEnd = tauStart + tau_max;

        const argsObj = {
            L: this.L, E: this.E, r_escape: this.#r_escape, atol: this.#atol, rtol: this.#rtol,
            max_step: (this.#max_step_factor ? Math.max(1e-12, tau_max * this.#max_step_factor) : Infinity),
            maxPoints: MAX_GEODESIC_POINTS
        };
        const fwrap = (tau, y) => geodesic_tau(tau, y, this.L, this.E);

        const res = DOPRI.integrate(fwrap, tauStart, y0, tauEnd, argsObj);
        if (res.tau.length === 0) return;

        const startIndex = (this.#tau_vals.length > 0 && Math.abs(res.tau[0] - tauStart) < 1e-14) ? 1 : 0;
        if (startIndex >= res.tau.length) return;

        for (let i = startIndex; i < res.tau.length; i++) {
            this.#tau_vals.push(res.tau[i]);
            const y = res.yout[i];
            this.#r_tau.push(y[0]);
            this.#ur_tau.push(y[1]);
            this.#phi_tau.push(y[2]);
            this.#t_tau.push(y[3]);
        }
        
        if (res.event) {
            this.alive = false;
            this.state = res.event === "horizon" ? "falling" : "escaping";
        }
        
        this.#enforce_monotone_t();
        this.#update_t_to_tau_monotone();
    }
    
    #enforce_monotone_t() {
        if (this.#t_tau.length < 2) return;
        for (let i = 0; i < this.#t_tau.length - 1; i++) {
            if (this.#t_tau[i+1] <= this.#t_tau[i]) {
                this.#tau_vals.length = i + 1;
                this.#r_tau.length = i + 1;
                this.#ur_tau.length = i + 1;
                this.#phi_tau.length = i + 1;
                this.#t_tau.length = i + 1;
                break;
            }
        }
    }

    #update_t_to_tau_monotone() {
        this.#t_to_tau_monotone = this.#t_tau.length > 1 && this.#t_tau.every((v, i, arr) => i === 0 || v > arr[i-1]);
    }

    #inv_ttau(t_rel) {
        if (!this.#t_to_tau_monotone || this.#t_tau.length < 2) return this.#tau_vals[0] ?? 0;
        
        let lo = 0, hi = this.#t_tau.length - 1;
        if (t_rel <= this.#t_tau[0]) return this.#tau_vals[0];
        if (t_rel >= this.#t_tau[hi]) return this.#tau_vals[hi];

        while (hi - lo > 1) {
            const mid = Math.floor((lo + hi) / 2);
            if (this.#t_tau[mid] <= t_rel) lo = mid; else hi = mid;
        }
        const t0 = this.#t_tau[lo], t1 = this.#t_tau[hi];
        const tau0 = this.#tau_vals[lo], tau1 = this.#tau_vals[hi];
        const frac = (t1 === t0) ? 0 : (t_rel - t0) / (t1 - t0);
        return tau0 + (tau1 - tau0) * frac;
    }

    #interp1(xs, ys, x) {
        if (xs.length === 0) return NaN;
        if (x <= xs[0]) return ys[0];
        if (x >= xs.at(-1)) return ys.at(-1);

        let lo = 0, hi = xs.length - 1;
        while (hi - lo > 1) {
            const mid = Math.floor((lo + hi) / 2);
            if (xs[mid] <= x) lo = mid; else hi = mid;
        }
        const x0 = xs[lo], x1 = xs[hi];
        const y0 = ys[lo], y1 = ys[hi];
        const frac = (x1 === x0) ? 0 : (x - x0) / (x1 - x0);
        return y0 + (y1 - y0) * frac;
    }

    resample_observer_time(t_common) {
        const n = t_common.length;
        this.x_obs = new Array(n).fill(NaN);
        this.y_obs = new Array(n).fill(NaN);
        if (this.#t_tau.length === 0) return;

        const t_start = this.#t_tau[0];
        const t_end = this.#t_tau.at(-1);

        for (let i = 0; i < n; i++) {
            const t_rel = t_common[i] - this.t_inject;
            if (t_rel >= t_start && t_rel <= t_end) {
                const tauObs = this.#inv_ttau(t_rel);
                const r_obs = this.#interp1(this.#tau_vals, this.#r_tau, tauObs);
                const phi_obs = this.#interp1(this.#tau_vals, this.#phi_tau, tauObs);
                this.x_obs[i] = r_obs * Math.cos(phi_obs);
                this.y_obs[i] = r_obs * Math.sin(phi_obs);
            }
        }
    }
    
    ensure_tau_covers(t_common, required_index) {
        if (!this.alive) return;
        const t_needed = t_common[required_index];
        const t_rel_needed = t_needed - this.t_inject;
        if (t_rel_needed <= 0) return;

        let iterations = 0;
        while ((this.#t_tau.length === 0 || t_rel_needed > this.#t_tau.at(-1)) && iterations < 50) {
            const last_len = this.#tau_vals.length;
            this.#extend_geodesic(TAU_CHUNK);
            iterations++;
            if (this.#tau_vals.length === last_len) break; // Integration stalled
        }
    }

    current_xyv(t_now, idx_hint) {
        if (!this.x_obs || this.x_obs.length === 0) {
            return [this.x0, this.y0, 0.0, 0.0];
        }

        let last_valid_idx = -1;
        for (let i = Math.min(idx_hint, this.x_obs.length - 1); i >= 0; i--) {
            if (Number.isFinite(this.x_obs[i])) {
                last_valid_idx = i;
                break;
            }
        }

        if (last_valid_idx < 0) return [NaN, NaN, NaN, NaN];

        const x = this.x_obs[last_valid_idx];
        const y = this.y_obs[last_valid_idx];

        let t_rel = t_now - this.t_inject;
        t_rel = clamp(t_rel, this.#t_tau[0], this.#t_tau.at(-1));
        
        const tau = this.#inv_ttau(t_rel);
        const r = this.#interp1(this.#tau_vals, this.#r_tau, tau);
        const ur = this.#interp1(this.#tau_vals, this.#ur_tau, tau);
        const phi = this.#interp1(this.#tau_vals, this.#phi_tau, tau);
        
        const f = 1.0 - Rg / Math.max(r, 1e-30);
        const f_safe = Math.max(f, F_FLOOR);
        const dt_dtau = this.E / f_safe;
        const dr_dt = ur / dt_dtau;
        const r_dphi_dt = (this.L / Math.max(r, 1e-30)) / dt_dtau;

        const vx = dr_dt * Math.cos(phi) - r_dphi_dt * Math.sin(phi);
        const vy = dr_dt * Math.sin(phi) + r_dphi_dt * Math.cos(phi);
        const v = Math.hypot(vx, vy);
        const theta = Math.atan2(vy, vx);

        return [x, y, v, theta];
    }
}


// ============================================================
// File: simulation.js
// ============================================================
class Simulation {
    #ui;
    #particles = [];
    #particle_counter = 0;
    #current_frame = 0;
    #playing = true;
    #tickHandle = null;

    #initial_T_total;
    #initial_nframes;
    #t_common;
    #dt;
    #r_escape;
    
    constructor(ui, params = {}) {
        this.#ui = ui;
        this.#initial_T_total = params.T_total ?? DEFAULT_TTOTAL;
        this.#initial_nframes = params.nframes ?? DEFAULT_NFRAMES;
        this.#r_escape = params.r_escape ?? DEFAULT_R_ESCAPE;
        
        this.#reset_time();
        this.#ui.bind_events(this);
        this.#ui.update_status("Status: ready");
    }

    start() {
        if (this.#tickHandle) clearInterval(this.#tickHandle);
        this.#tickHandle = setInterval(() => this.#updateFrame(), ANIMATION_INTERVAL_MS);
        this.#playing = true;
        this.#ui.update_play_button(this.#playing);
    }
    
    #reset_time() {
        this.#t_common = linspace(0.0, this.#initial_T_total, this.#initial_nframes);
        this.#dt = (this.#t_common.length > 1) ? (this.#t_common[1] - this.#t_common[0]) : 1.0;
    }

    #updateFrame() {
        this.#current_frame++;
        if (this.#current_frame >= this.#t_common.length - 10) {
            this.#extend_t_common();
        }
        
        const t_now = this.#t_common[Math.min(this.#current_frame, this.#t_common.length - 1)];

        for (const p of this.#particles) {
            if (p.alive && (p.t_inject <= t_now)) {
                p.ensure_tau_covers(this.#t_common, this.#current_frame);
                p.resample_observer_time(this.#t_common);
            }
        }
        
        this.#ui.render(this.#particles, t_now, this.#current_frame);
    }
    
    #extend_t_common(n_extra_frames = null) {
        if (n_extra_frames === null) n_extra_frames = this.#initial_nframes;
        const last_t = this.#t_common.at(-1);
        const new_times = new Array(n_extra_frames);
        for (let i = 0; i < n_extra_frames; i++) {
            new_times[i] = last_t + this.#dt * (i + 1);
        }
        this.#t_common.push(...new_times);
    }

    // --- Public methods for UI interaction ---
    addParticle(x, y, theta, v) {
        const vx = v * Math.cos(theta);
        const vy = v * Math.sin(theta);
        const t_inj = this.#t_common[Math.min(this.#current_frame, this.#t_common.length - 1)];
        const color = `hsl(${(this.#particle_counter * 37) % 360} 80% 45%)`;
        const name = `P${String(this.#particle_counter + 1).padStart(3, '0')}`;
        
        try {
            const p = new Particle(x, y, vx, vy, t_inj, { name, color, r_escape: this.#r_escape });
            p.resample_observer_time(this.#t_common);
            this.#particles.push(p);
            this.#particle_counter++;
            this.#ui.add_particle_to_table(p);
        } catch (e) {
            console.error("[particle creation failed]", e);
            this.#ui.update_status(`Error: ${e.message}`);
        }
    }

    reset() {
        this.#particles = [];
        this.#particle_counter = 0;
        this.#current_frame = 0;
        this.#reset_time();
        this.#ui.reset(this.#particles);
        this.#ui.update_status("Status: reset");
    }

    togglePlay() {
        if (this.#playing) {
            this.#playing = false;
            clearInterval(this.#tickHandle);
            this.#tickHandle = null;
        } else {
            this.#playing = true;
            this.start();
        }
        this.#ui.update_play_button(this.#playing);
    }

    isPlaying() { return this.#playing; }
}


// ============================================================
// File: ui.js
// (Handles all DOM and Canvas interactions)
// ============================================================
class UI {
    #dom;
    #canvas; #ctx; #W; #H;
    #scale = 1.0; #origin = [0, 0]; #R_display = 80.0;
    #show_trails = true; #auto_zoom = true;
    #highlighted = new Set();
    #table_row_elements = new Map(); // For optimized table updates

    constructor() {
        this.#dom = {
            sliderX: document.getElementById("sliderX"),
            sliderY: document.getElementById("sliderY"),
            sliderTheta: document.getElementById("sliderTheta"),
            sliderV: document.getElementById("sliderV"),
            valX: document.getElementById("valX"),
            valY: document.getElementById("valY"),
            valTheta: document.getElementById("valTheta"),
            valV: document.getElementById("valV"),
            btnAdd: document.getElementById("btnAdd"),
            btnReset: document.getElementById("btnReset"),
            btnTrails: document.getElementById("btnTrails"),
            btnPlay: document.getElementById("btnPlay"),
            tableBody: document.getElementById("particlesBody"),
            status: document.getElementById("status"),
            zoomSlider: document.getElementById("zoomSlider"),
            zoomVal: document.getElementById("zoomVal"),
            observerTime: document.getElementById("observerTime"),
        };
        this.#canvas = document.getElementById("simCanvas");
        this.#ctx = this.#canvas.getContext("2d");
        this.#fitCanvas();
        window.addEventListener("resize", () => this.#fitCanvas());
    }

    bind_events(controller) {
        // Sliders
        ['sliderX', 'sliderY', 'sliderTheta', 'sliderV'].forEach(id => {
            this.#dom[id].oninput = () => this.#update_preview_values();
        });
        
        // Buttons
        this.#dom.btnAdd.onclick = () => {
            const vals = this.#get_control_values();
            controller.addParticle(vals.x, vals.y, vals.theta, vals.v);
        };
        this.#dom.btnReset.onclick = () => controller.reset();
        this.#dom.btnPlay.onclick = () => controller.togglePlay();
        this.#dom.btnTrails.onclick = () => {
            this.#show_trails = !this.#show_trails;
            this.update_status(`Trails: ${this.#show_trails ? "ON" : "OFF"}`);
        };
        
        // Zoom
        this.#dom.zoomSlider.oninput = () => {
            this.#auto_zoom = false;
            this.#setZoom(Number(this.#dom.zoomSlider.value));
        };
        
        // Up/Down buttons for sliders
        document.querySelectorAll(".btn-up, .btn-down").forEach(btn => {
            let holdTimer;
            const stepSlider = () => {
                const slider = document.getElementById(btn.dataset.target);
                if (!slider) return;
                const step = Number(slider.step) || 1;
                let val = Number(slider.value) + (btn.classList.contains("btn-up") ? step : -step);
                slider.value = clamp(val, Number(slider.min), Number(slider.max));
                slider.dispatchEvent(new Event("input"));
            };
            btn.addEventListener("click", stepSlider);
            btn.addEventListener("mousedown", () => { holdTimer = setInterval(stepSlider, 100); });
            btn.addEventListener("touchstart", (e) => { e.preventDefault(); holdTimer = setInterval(stepSlider, 100); });
            ["mouseup", "mouseleave", "touchend", "touchcancel"].forEach(ev =>
                btn.addEventListener(ev, () => clearInterval(holdTimer))
            );
        });
    }

    render(particles, t_now, frame_idx) {
        this.#clearCanvas();
        this.#drawBackground();
        this.#drawTrailsAndPoints(particles, frame_idx);
        this.#drawPreview();
        this.#updateTable(particles, t_now, frame_idx);
        this.#dom.observerTime.textContent = `Observer Time: ${t_now.toFixed(3)}`;
    }
    
    reset(particles) {
        this.#table_row_elements.clear();
        this.#dom.tableBody.innerHTML = '';
        this.#highlighted.clear();
        this.#R_display = 80.0;
        this.#dom.zoomSlider.value = this.#R_display;
        this.#dom.zoomVal.textContent = this.#R_display;
        this.render(particles, 0, 0); // Initial render
    }
    
    // --- Optimized Table Update ---
    add_particle_to_table(particle) {
        const tr = document.createElement("tr");

        const chk_td = document.createElement("td");
        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.onchange = () => {
            if (chk.checked) this.#highlighted.add(particle.name);
            else this.#highlighted.delete(particle.name);
        };
        chk_td.appendChild(chk);
        
        const name_td = document.createElement("td"); name_td.textContent = particle.name;
        const x_td = document.createElement("td");
        const y_td = document.createElement("td");
        const theta_td = document.createElement("td");
        const v_td = document.createElement("td");
        const E_td = document.createElement("td"); E_td.textContent = particle.E.toPrecision(6);
        const L_td = document.createElement("td"); L_td.textContent = particle.L.toPrecision(6);
        const state_td = document.createElement("td");
        
        tr.append(chk_td, name_td, x_td, y_td, theta_td, v_td, E_td, L_td, state_td);
        this.#dom.tableBody.appendChild(tr);

        this.#table_row_elements.set(particle.name, { x_td, y_td, theta_td, v_td, state_td });
    }

    #updateTable(particles, t_now, frame_idx) {
        for (const p of particles) {
            const row = this.#table_row_elements.get(p.name);
            if (!row) continue;
            
            const state = this.#get_particle_state_string(p, t_now, frame_idx);
            row.state_td.textContent = state;

            if (p.state === "escaping") {
                row.x_td.textContent = "--"; row.y_td.textContent = "--";
                row.v_td.textContent = "--"; row.theta_td.textContent = "--";
            } else {
                const [x, y, v, theta] = p.current_xyv(t_now, frame_idx);
                row.x_td.textContent = Number.isFinite(x) ? x.toPrecision(6) : "N/A";
                row.y_td.textContent = Number.isFinite(y) ? y.toPrecision(6) : "N/A";
                row.v_td.textContent = Number.isFinite(v) ? v.toPrecision(6) : "N/A";
                row.theta_td.textContent = Number.isFinite(theta) ? theta.toFixed(6) : "N/A";
            }
        }
    }
    
    #get_particle_state_string(p, t_now, idx) {
        if (t_now < p.t_inject - 1e-12) return "pending";
        if (!p.alive) return p.state;
        
        const upto = Math.min(idx, (p.x_obs?.length ?? 0) -1);
        let valid = false;
        for (let i = 0; i <= upto; i++) if (Number.isFinite(p.x_obs[i])) { valid = true; break; }
        if (valid) return "active";
        
        return "integrating";
    }

    // --- Canvas and Drawing ---
    #fitCanvas() {
        const rect = this.#canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const size = Math.min(rect.width, rect.height);
        this.#canvas.style.width  = `${size}px`;
        this.#canvas.style.height = `${size}px`;
        this.#canvas.width  = Math.round(size * dpr); 
        this.#canvas.height = Math.round(size * dpr);
        this.#W = this.#canvas.width;
        this.#H = this.#canvas.height;
        this.#origin = [this.#W / 2, this.#H / 2];
        this.#setZoom(this.#R_display, true);
    }
    
    #setZoom(R_display, force = false) {
        this.#R_display = R_display;
        this.#scale = Math.min(this.#W, this.#H) / (2 * this.#R_display);
        if (force) return;
        this.#dom.zoomSlider.value = this.#R_display;
        this.#dom.zoomVal.textContent = this.#R_display;
    }

    #worldToCanvas(x, y) {
        return [this.#origin[0] + x * this.#scale, this.#origin[1] - y * this.#scale];
    }

    #clearCanvas() { this.#ctx.clearRect(0, 0, this.#W, this.#H); }
    
    #drawBackground() {
        const [cx, cy] = this.#worldToCanvas(0, 0);
        const pxR = Math.max(1, Rg * this.#scale);
        this.#ctx.beginPath();
        this.#ctx.fillStyle = "#000";
        this.#ctx.arc(cx, cy, pxR, 0, Math.PI*2);
        this.#ctx.fill();
        this.#ctx.lineWidth = 1;
        this.#ctx.strokeStyle = "#555";
        this.#ctx.stroke();
    }
    
    #drawTrailsAndPoints(particles, frame) {
        for (const p of particles) {
            const is_hl = this.#highlighted.has(p.name);
            if (this.#show_trails && p.x_obs) {
                this.#ctx.beginPath();
                let started = false;
                for (let j = 0; j <= Math.min(frame, p.x_obs.length - 1); j++) {
                    if (!Number.isFinite(p.x_obs[j])) { started = false; continue; }
                    const [px, py] = this.#worldToCanvas(p.x_obs[j], p.y_obs[j]);
                    if (!started) { this.#ctx.moveTo(px, py); started = true; }
                    else { this.#ctx.lineTo(px, py); }
                }
                this.#ctx.lineWidth = is_hl ? 4 : 2;
                this.#ctx.strokeStyle = is_hl ? "gold" : p.color;
                this.#ctx.globalAlpha = is_hl ? 1.0 : 0.6;
                this.#ctx.stroke();
            }
            
            const xnow = p.x_obs?.[frame];
            const ynow = p.y_obs?.[frame];
            if (Number.isFinite(xnow)) {
                const [px, py] = this.#worldToCanvas(xnow, ynow);
                this.#ctx.globalAlpha = 1.0;
                this.#ctx.beginPath();
                this.#ctx.fillStyle = is_hl ? "gold" : p.color;
                this.#ctx.arc(px, py, is_hl ? 8 : 5, 0, Math.PI*2);
                this.#ctx.fill();
                
                if (this.#auto_zoom && (Math.abs(xnow) > 0.95 * this.#R_display || Math.abs(ynow) > 0.95 * this.#R_display)) {
                    this.#setZoom(this.#R_display * 1.5);
                }
            }
        }
        this.#ctx.globalAlpha = 1.0;
    }

    #drawPreview() {
        const {x, y, theta, v} = this.#get_control_values();
        const [px, py] = this.#worldToCanvas(x, y);
        this.#ctx.beginPath();
        this.#ctx.fillStyle = "red";
        this.#ctx.arc(px, py, 6, 0, Math.PI*2);
        this.#ctx.fill();

        if (v > 0) {
            const arrowLen = v * this.#R_display / 8 + this.#R_display / 18;
            const dx = arrowLen * Math.cos(theta);
            const dy = arrowLen * Math.sin(theta);
            const [ex, ey] = this.#worldToCanvas(x + dx, y + dy);
            this.#drawArrow(px, py, ex, ey, "red");
        }
    }

    #drawArrow(sx, sy, ex, ey, color) {
        this.#ctx.strokeStyle = color;
        this.#ctx.fillStyle = color;
        this.#ctx.lineWidth = 2;
        this.#ctx.beginPath();
        this.#ctx.moveTo(sx, sy);
        this.#ctx.lineTo(ex, ey);
        this.#ctx.stroke();
        const ang = Math.atan2(ey - sy, ex - sx);
        const size = 8 * (this.#W / 800); // Scale arrow head with canvas size
        this.#ctx.beginPath();
        this.#ctx.moveTo(ex, ey);
        this.#ctx.lineTo(ex - size * Math.cos(ang - Math.PI/6), ey - size * Math.sin(ang - Math.PI/6));
        this.#ctx.lineTo(ex - size * Math.cos(ang + Math.PI/6), ey - size * Math.sin(ang + Math.PI/6));
        this.#ctx.closePath();
        this.#ctx.fill();
    }

    // --- UI State & Value Getters/Setters ---
    #update_preview_values() {
        const { x, y, theta, v } = this.#get_control_values();
        this.#dom.valX.textContent = x.toFixed(2);
        this.#dom.valY.textContent = y.toFixed(2);
        this.#dom.valTheta.textContent = theta.toFixed(2);
        this.#dom.valV.textContent = v.toFixed(3);
        this.#drawPreview(); // Re-render preview overlay only
    }

    #get_control_values() {
        return {
            x: Number(this.#dom.sliderX.value),
            y: Number(this.#dom.sliderY.value),
            theta: Number(this.#dom.sliderTheta.value),
            v: Number(this.#dom.sliderV.value)
        };
    }
    
    update_play_button(playing) {
        this.#dom.btnPlay.textContent = playing ? "Pause" : "Play";
    }

    update_status(message) {
        this.#dom.status.textContent = message;
    }
}


// ============================================================
// File: main.js
// (Application Entry Point)
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    const ui = new UI();
    const sim = new Simulation(ui, { T_total: DEFAULT_TTOTAL, nframes: DEFAULT_NFRAMES });
    sim.start();
});