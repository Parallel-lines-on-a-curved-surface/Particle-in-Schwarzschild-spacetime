/* -------------------------------------------------------------
   Schwarzschild Geodesic Simulation (JS)
   - faithful translation of Python logic into browser JS
   - adaptive Dormand-Prince RK45 integrator implemented
   - UI: sliders, buttons, table, canvas
   ------------------------------------------------------------- */

/* ---------------- Constants (match Python file) ---------------- */
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

/* ---------------- utility math helpers ---------------- */
function linspace(a, b, n) {
  if (n <= 1) return [a];
  const out = new Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; ++i) out[i] = a + step * i;
  return out;
}
function hypot(x, y) { return Math.hypot(x, y); }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function isFiniteArr(x) { return Number.isFinite(x); }
function approxEqual(a,b,tol=1e-12) { return Math.abs(a-b) <= tol; }

/* ---------------- metric and geodesic rhs ---------------- */
function metricElements(r) {
  const f = 1.0 - Rg / r;
  return {f: f, g_tt: -f, g_rr: 1.0 / f, g_pp: r*r};
}
// y = [r, ur, phi, tcoord]
// returns dy/dtau
function geodesic_tau(tau, y, L, E) {
  const r = y[0], ur = y[1], phi = y[2], tcoord = y[3];
  const f = 1.0 - Rg / r;
  // d2r  (using same formula as Python port)
  const d2r = (Rg / (2.0 * f * r*r)) * (- E*E + ur*ur) + f * (L*L) / (r*r*r);
  const dphi = L / (r*r);
  const dt = E / f;
  return [ur, d2r, dphi, dt];
}

/* ---------------- Dormand-Prince RK45 integrator ----------------
   Implemented with adaptive step control (coeffs from DOPRI5)
   Integrate vector ODE dy/dtau = f(tau,y)
   Returns arrays tau_arr, y_arr (each y is array length 4)
-----------------------------------------------------------------*/
const DOPRI = (function(){
  // coeffs
  const c2 = 1/5, c3 = 3/10, c4 = 4/5, c5 = 8/9, c6 = 1.0, c7 = 1.0;
  const a = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
  ];
  const b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0];
  const b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40];
  // helpers: vector ops on length-4 arrays
  function addScaled(y, ks, coeffs, h) {
    // y + h * sum_i coeffs[i] * ks[i]
    const out = [y[0], y[1], y[2], y[3]];
    for (let i=0;i<coeffs.length;i++) {
      const c = coeffs[i] * h;
      const k = ks[i];
      out[0] += c * k[0];
      out[1] += c * k[1];
      out[2] += c * k[2];
      out[3] += c * k[3];
    }
    return out;
  }
  function vecAdd(a,b,scale=1.0) {
    return [a[0]+scale*b[0], a[1]+scale*b[1], a[2]+scale*b[2], a[3]+scale*b[3]];
  }
  function vecScale(y,s) { return [y[0]*s,y[1]*s,y[2]*s,y[3]*s]; }
  function absMax(arr) { return Math.max(Math.abs(arr[0]),Math.abs(arr[1]),Math.abs(arr[2]),Math.abs(arr[3])); }

  // one adaptive integrate chunk: from tau0 to tau1 (tau1 > tau0), accumulate accepted points
  function integrate(f, tau0, y0, tau1, argsObj) {
    // argsObj: { L, E, r_escape, atol, rtol, max_step, maxPoints }
    const atol = argsObj.atol ?? DEFAULT_ATOL;
    const rtol = argsObj.rtol ?? DEFAULT_RTOL;
    const r_escape = argsObj.r_escape ?? DEFAULT_R_ESCAPE;
    const max_step = argsObj.max_step ?? Infinity;
    const maxPoints = argsObj.maxPoints ?? 1000000;

    const outTau = [];
    const outY = [];
    let tau = tau0;
    let y = [y0[0], y0[1], y0[2], y0[3]];
    // initial step guess:
    let h = Math.min(1.0, (tau1 - tau0) / 100.0);
    h = Math.min(h, max_step);
    const min_h = 1e-12;
    const maxIter = 1000000;
    let iter = 0;

    while (tau < tau1 - 1e-15 && iter < maxIter && outTau.length < maxPoints) {
      iter++;
      if (tau + h > tau1) h = tau1 - tau;
      // compute k1..k7
      const k1 = f(tau, y);
      const k2 = f(tau + c2*h, addScaled(y, [k1], [a[1][0]], h));
      const k3 = f(tau + c3*h, addScaled(y, [k1,k2], [a[2][0], a[2][1]], h));
      const k4 = f(tau + c4*h, addScaled(y, [k1,k2,k3], [a[3][0], a[3][1], a[3][2]], h));
      const k5 = f(tau + c5*h, addScaled(y, [k1,k2,k3,k4], [a[4][0], a[4][1], a[4][2], a[4][3]], h));
      const k6 = f(tau + c6*h, addScaled(y, [k1,k2,k3,k4,k5], [a[5][0], a[5][1], a[5][2], a[5][3], a[5][4]], h));
      const k7 = f(tau + c7*h, addScaled(y, [k1,k2,k3,k4,k5,k6], [a[6][0], a[6][1], a[6][2], a[6][3], a[6][4], a[6][5]], h));
      // y5 and y4
      const y5 = [
        y[0] + h*(b5[0]*k1[0] + b5[1]*k2[0] + b5[2]*k3[0] + b5[3]*k4[0] + b5[4]*k5[0] + b5[5]*k6[0] + b5[6]*k7[0]),
        y[1] + h*(b5[0]*k1[1] + b5[1]*k2[1] + b5[2]*k3[1] + b5[3]*k4[1] + b5[4]*k5[1] + b5[5]*k6[1] + b5[6]*k7[1]),
        y[2] + h*(b5[0]*k1[2] + b5[1]*k2[2] + b5[2]*k3[2] + b5[3]*k4[2] + b5[4]*k5[2] + b5[5]*k6[2] + b5[6]*k7[2]),
        y[3] + h*(b5[0]*k1[3] + b5[1]*k2[3] + b5[2]*k3[3] + b5[3]*k4[3] + b5[4]*k5[3] + b5[5]*k6[3] + b5[6]*k7[3])
      ];
      const y4 = [
        y[0] + h*(b4[0]*k1[0] + b4[1]*k2[0] + b4[2]*k3[0] + b4[3]*k4[0] + b4[4]*k5[0] + b4[5]*k6[0] + b4[6]*k7[0]),
        y[1] + h*(b4[0]*k1[1] + b4[1]*k2[1] + b4[2]*k3[1] + b4[3]*k4[1] + b4[4]*k5[1] + b4[5]*k6[1] + b4[6]*k7[1]),
        y[2] + h*(b4[0]*k1[2] + b4[1]*k2[2] + b4[2]*k3[2] + b4[3]*k4[2] + b4[4]*k5[2] + b4[5]*k6[2] + b4[6]*k7[2]),
        y[3] + h*(b4[0]*k1[3] + b4[1]*k2[3] + b4[2]*k3[3] + b4[3]*k4[3] + b4[4]*k5[3] + b4[5]*k6[3] + b4[6]*k7[3])
      ];
      // error estimate
      const err = [y5[0]-y4[0], y5[1]-y4[1], y5[2]-y4[2], y5[3]-y4[3]];
      // compute norm
      // Use component-wise scaling (atol + rtol * max(|y|,|y5|))
      let errNorm = 0;
      for (let i=0;i<4;i++) {
        const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(y5[i]));
        const comp = Math.abs(err[i]) / sc;
        if (comp > errNorm) errNorm = comp;
      }
      // adapt step
      if (errNorm <= 1.0) {
        // accept
        tau += h;
        y = y5;
        outTau.push(tau);
        outY.push([y[0], y[1], y[2], y[3]]);
        // adjust h
        const safety = 0.9;
        const exp = 1.0/5.0;
        const factor = Math.max(0.2, Math.min(5.0, safety * Math.pow(1/Math.max(errNorm,1e-16), exp)));
        h = Math.min(h * factor, max_step);
        if (h < min_h) h = min_h;
        // event detection: horizon or escape if r crosses thresholds between last recorded out and current y
        const rnow = outY.length>=2 ? outY[outY.length-2][0] : null; // approximate
        // We'll do simple check: if current y[0] <= HORIZON_MARGIN*Rg -> return with event
        if (y[0] <= HORIZON_MARGIN * Rg) {
          return {tau: outTau, yout: outY, event: "horizon", terminated: true};
        }
        if (y[0] >= r_escape) {
          return {tau: outTau, yout: outY, event: "escape", terminated: true};
        }
      } else {
        // reject and shrink h
        const safety = 0.9;
        const exp = 1.0/5.0;
        const factor = Math.max(0.1, safety * Math.pow(1/errNorm, exp));
        h *= factor;
        if (h < min_h) h = min_h;
      }
    } // end while
    return {tau: outTau, yout: outY, event: null, terminated: false};
  }

  return { integrate };
})();

/* ---------------- Particle class (JS port) ---------------- */
class Particle {
  constructor(x0,y0,vx0,vy0,t_inject=0.0, opts={}) {
    this.x0 = Number(x0); this.y0 = Number(y0);
    this.vx0 = Number(vx0); this.vy0 = Number(vy0);
    this.t_inject = Number(t_inject);
    this.rtol = opts.rtol ?? DEFAULT_RTOL;
    this.atol = opts.atol ?? DEFAULT_ATOL;
    this.r_escape = opts.r_escape ?? DEFAULT_R_ESCAPE;
    this.name = opts.name ?? "Noname";
    this.color = opts.color ?? "steelblue";
    this.max_step_factor = opts.max_step_factor ?? DEFAULT_MAX_STEP_FACTOR;
    this.method = "DOPRI5";

    // saved arrays (tau parameter)
    this.tau_vals = []; // proper time samples
    this.r_tau = [];
    this.ur_tau = [];
    this.phi_tau = [];
    this.t_tau = [];

    // observer-sampled positions
    this.x_obs = null;
    this.y_obs = null;

    this._t_to_tau_monotone = false;

    this.alive = true;
    this.state = "active";
    this.L = 0;
    this.E = 0;
    this.u0 = null;

    // compute initial conditions & extend
    this._compute_initial_conditions();
    this._extend_geodesic(TAU_CHUNK);
  }

  _compute_initial_conditions() {
    let x0=this.x0, y0=this.y0, vx0=this.vx0, vy0=this.vy0;
    let r0 = Math.hypot(x0,y0);
    let phi0 = Math.atan2(y0,x0);
    if (r0 <= HORIZON_MARGIN * Rg) {
      const r_old = r0;
      r0 = HORIZON_MARGIN*Rg * 1.0001;
      x0 = r0 * Math.cos(phi0);
      y0 = r0 * Math.sin(phi0);
      this.x0 = x0; this.y0 = y0;
      console.warn(`[warning] r0 too small (${r_old}); bumped to ${r0}`);
    }
    const {f,g_tt,g_rr,g_pp} = metricElements(r0);
    const v_r = Math.cos(phi0)*vx0 + Math.sin(phi0)*vy0;
    const v_tang = -Math.sin(phi0)*vx0 + Math.cos(phi0)*vy0;
    const dphi_dt = (r0 !== 0 ? v_tang / r0 : 0.0);

    let denom = g_tt + g_rr * (v_r*v_r) + g_pp * (dphi_dt*dphi_dt);
    if (!Number.isFinite(denom)) {
      throw new Error(`[Particle ${this.name}] non-finite denom in initial condition: ${denom}`);
    }
    if (denom >= 0.0) {
      let scale = 0.999;
      let v_r_s = v_r * scale;
      let dphi_dt_s = dphi_dt * scale;
      denom = g_tt + g_rr * (v_r_s*v_r_s) + g_pp * (dphi_dt_s*dphi_dt_s);
      if (denom >= 0.0) {
        throw new Error(`[${this.name}] invalid initial velocity: cannot form timelike 4-velocity (denom=${denom})`);
      }
    }
    const u_t = Math.sqrt(-1.0 / denom);
    const u_r = u_t * v_r;
    const u_phi = u_t * dphi_dt;
    const E = -g_tt * u_t;
    const L = g_pp * u_phi;
    this.y_init = [r0, u_r, phi0, 0.0];
    this.L = Number(L);
    this.E = Number(E);
    this.u0 = [u_t, u_r, u_phi];
    // normalization residual check
    const contraction = g_tt*u_t*u_t + g_rr*u_r*u_r + g_pp*u_phi*u_phi;
    if (!Number.isFinite(contraction) || Math.abs(contraction + 1.0) > 1e-8) {
      console.info(`[info] normalization residual: ${(contraction+1.0)}`);
    }
  }

  _extend_geodesic(tau_max=TAU_CHUNK) {
    if (!this.alive) return;
    let y0, tauStart;
    if (this.tau_vals.length > 0) {
      // estimate ur from last two if available
      let ur_guess = 0.0;
      const n = this.tau_vals.length;
      if (n >= 2) {
        ur_guess = (this.r_tau[n-1] - this.r_tau[n-2]) / (this.tau_vals[n-1] - this.tau_vals[n-2]);
      }
      y0 = [this.r_tau[this.r_tau.length-1], ur_guess, this.phi_tau[this.phi_tau.length-1], this.t_tau[this.t_tau.length-1]];
      tauStart = this.tau_vals[this.tau_vals.length-1];
    } else {
      y0 = this.y_init.slice();
      tauStart = 0.0;
    }
    const tauEnd = tauStart + tau_max;
    // prepare integrator args
    const argsObj = {L: this.L, E: this.E, r_escape: this.r_escape, atol: this.atol, rtol: this.rtol, max_step: (this.max_step_factor?Math.max(1e-12, tau_max*this.max_step_factor):Infinity), maxPoints: MAX_GEODESIC_POINTS};
    // function f for DOPRI: wrap to pass L,E captured via closure; geodesic_tau conforms
    const fwrap = (tau, y) => geodesic_tau(tau, y, this.L, this.E);
    const res = DOPRI.integrate(fwrap, tauStart, y0, tauEnd, argsObj);
    // res.tau is array of tau (absolute), res.yout is array of y states
    if (res.tau.length === 0) {
      // nothing appended
      return;
    }
    // append, but if initial overlap remove duplicate first if matches tauStart
    if (this.tau_vals.length === 0) {
      this.tau_vals = res.tau.slice();
      this.r_tau = res.yout.map(y => y[0]);
      this.ur_tau = res.yout.map(y => y[1]);
      this.phi_tau = res.yout.map(y => y[2]);
      this.t_tau = res.yout.map(y => y[3]);
    } else {
      // avoid duplicate tau equal to last
      let t_new = res.tau.slice();
      let y_new = res.yout.slice();
      if (Math.abs(t_new[0] - this.tau_vals[this.tau_vals.length-1]) < 1e-14) {
        t_new.shift(); y_new.shift();
      }
      if (t_new.length > 0) {
        this.tau_vals = this.tau_vals.concat(t_new);
        this.r_tau = this.r_tau.concat(y_new.map(y => y[0]));
        this.ur_tau = this.ur_tau.concat(y_new.map(y => y[1]));
        this.phi_tau = this.phi_tau.concat(y_new.map(y => y[2]));
        this.t_tau = this.t_tau.concat(y_new.map(y => y[3]));
      }
    }
    // check events
    if (res.event === "horizon") { this.alive = false; this.state = "falling"; }
    else if (res.event === "escape") { this.alive = false; this.state = "escaping"; }

    this._enforce_monotone_t();
    this._maybe_decimate();
    this._update_t_to_tau_monotone();
  }

  _enforce_monotone_t() {
    if (this.t_tau.length < 2) return;
    // remove non-finite
    const goodIdx = [];
    for (let i=0;i<this.t_tau.length;i++) if (Number.isFinite(this.t_tau[i])) goodIdx.push(i);
    if (goodIdx.length !== this.t_tau.length) {
      this.tau_vals = goodIdx.map(i => this.tau_vals[i]);
      this.r_tau = goodIdx.map(i => this.r_tau[i]);
      this.ur_tau = goodIdx.map(i => this.ur_tau[i]);
      this.phi_tau = goodIdx.map(i => this.phi_tau[i]);
      this.t_tau = goodIdx.map(i => this.t_tau[i]);
    }
    for (let i=0;i<this.t_tau.length-1;i++) {
      if (this.t_tau[i+1] <= this.t_tau[i]) {
        // cut at i+1
        this.tau_vals = this.tau_vals.slice(0, i+1);
        this.r_tau = this.r_tau.slice(0, i+1);
        this.ur_tau = this.ur_tau.slice(0, i+1);
        this.phi_tau = this.phi_tau.slice(0, i+1);
        this.t_tau = this.t_tau.slice(0, i+1);
        break;
      }
    }
  }

  _maybe_decimate(max_points=MAX_GEODESIC_POINTS) {
    const n = this.tau_vals.length;
    if (n > max_points) {
      const k = Math.ceil(n / max_points);
      const idx = [];
      for (let i=0;i<n-1;i+=k) idx.push(i);
      idx.push(n-1);
      this.tau_vals = idx.map(i => this.tau_vals[i]);
      this.r_tau = idx.map(i => this.r_tau[i]);
      this.ur_tau = idx.map(i => this.ur_tau[i]);
      this.phi_tau = idx.map(i => this.phi_tau[i]);
      this.t_tau = idx.map(i => this.t_tau[i]);
    }
  }

  _update_t_to_tau_monotone() {
    this._t_to_tau_monotone = (this.t_tau.length > 1) && this.t_tau.every((v,i,arr)=> i===0 || v>arr[i-1]);
  }

  // linear inverse interpolation: given t_rel find tau by searching t_tau (monotone)
  _inv_ttau(t_rel) {
    if (!this._t_to_tau_monotone) {
      // fallback: linear interpolation of arrays after rebuilding monotone mapping if possible
      // simple brute-force search
      let bestIdx = 0;
      for (let i=0;i<this.t_tau.length-1;i++) {
        if (this.t_tau[i] <= t_rel && t_rel <= this.t_tau[i+1]) { bestIdx = i; break; }
      }
      const t0 = this.t_tau[bestIdx], t1 = this.t_tau[bestIdx+1];
      const tau0 = this.tau_vals[bestIdx], tau1 = this.tau_vals[bestIdx+1];
      const frac = (t1===t0?0:(t_rel - t0)/(t1 - t0));
      return tau0 + (tau1 - tau0) * frac;
    }
    // binary search
    let lo = 0, hi = this.t_tau.length-1;
    if (t_rel <= this.t_tau[0]) return this.tau_vals[0];
    if (t_rel >= this.t_tau[hi]) return this.tau_vals[hi];
    while (hi - lo > 1) {
      const mid = Math.floor((lo+hi)/2);
      if (this.t_tau[mid] <= t_rel) lo = mid; else hi = mid;
    }
    const t0 = this.t_tau[lo], t1 = this.t_tau[hi];
    const tau0 = this.tau_vals[lo], tau1 = this.tau_vals[hi];
    const frac = (t1===t0?0:(t_rel - t0)/(t1 - t0));
    return tau0 + (tau1 - tau0) * frac;
  }

  resample_observer_time(t_common) {
    const n = t_common.length;
    this.x_obs = new Array(n).fill(NaN);
    this.y_obs = new Array(n).fill(NaN);
    if (this.t_tau.length === 0) return;
    const t_rel_arr = t_common.map(t => t - this.t_inject);
    // find valid indices
    for (let i=0;i<n;i++) {
      const tr = t_rel_arr[i];
      if (tr >= this.t_tau[0] && tr <= this.t_tau[this.t_tau.length-1]) {
        const tauObs = this._inv_ttau(tr);
        // linear interpolation in tau -> r, phi
        const r_obs = this._interp1(this.tau_vals, this.r_tau, tauObs);
        const phi_obs = this._interp1(this.tau_vals, this.phi_tau, tauObs);
        this.x_obs[i] = r_obs * Math.cos(phi_obs);
        this.y_obs[i] = r_obs * Math.sin(phi_obs);
      }
    }
  }

  _interp1(xs, ys, x) {
    if (xs.length === 0) return NaN;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length-1]) return ys[ys.length-1];
    // binary search
    let lo = 0, hi = xs.length-1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo+hi)/2);
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const x0 = xs[lo], x1 = xs[hi];
    const y0 = ys[lo], y1 = ys[hi];
    const frac = (x1===x0?0:(x - x0)/(x1 - x0));
    return y0 + (y1 - y0) * frac;
  }

  ensure_tau_covers(t_common, required_index) {
    if (!this.alive) return;
    const t_needed = t_common[required_index];
    const t_rel_needed = t_needed - this.t_inject;
    if (t_rel_needed <= 0) return;
    let iterations = 0;
    while ((this.t_tau.length === 0 || t_rel_needed > this.t_tau[this.t_tau.length-1]) && iterations < 50) {
      const last_len = this.tau_vals.length;
      this._extend_geodesic(TAU_CHUNK);
      iterations++;
      if (this.tau_vals.length === last_len) break;
    }
  }

  // returns x,y,v,theta as in Python current_xyv
  current_xyv(t_now, idx_hint) {
    // 1) use observed arrays if available
    if (this.x_obs && this.y_obs && this.x_obs.length > 0) {
      const upto = Math.min(idx_hint, this.x_obs.length - 1);
      let j = -1;
      for (let i=0;i<=upto;i++) if (Number.isFinite(this.x_obs[i]) && Number.isFinite(this.y_obs[i])) j = i;
      if (j >= 0) {
        const x = this.x_obs[j], y = this.y_obs[j];
        // compute local velocity components
        let t_rel = t_now - this.t_inject;
        t_rel = clamp(t_rel, this.t_tau[0], this.t_tau[this.t_tau.length-1]);
        const tau = this._inv_ttau(t_rel);
        const r = this._interp1(this.tau_vals, this.r_tau, tau);
        const ur = this._interp1(this.tau_vals, this.ur_tau, tau);
        const phi = this._interp1(this.tau_vals, this.phi_tau, tau);
        const f = 1.0 - Rg / Math.max(r, 1e-30);
        const f_safe = (f > F_FLOOR ? f : F_FLOOR);
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
    // fallback to last tau point
    if (this.r_tau.length > 0) {
      const r = this.r_tau[this.r_tau.length-1];
      const phi = this.phi_tau[this.phi_tau.length-1];
      const ur = this.ur_tau[this.ur_tau.length-1];
      const x = r * Math.cos(phi), y = r * Math.sin(phi);
      const f = 1.0 - Rg / Math.max(r, 1e-30);
      const f_safe = (f > F_FLOOR ? f : F_FLOOR);
      const dt_dtau = this.E / f_safe;
      const dr_dt = ur / dt_dtau;
      const r_dphi_dt = (this.L / Math.max(r, 1e-30)) / dt_dtau;
      const vx = dr_dt * Math.cos(phi) - r_dphi_dt * Math.sin(phi);
      const vy = dr_dt * Math.sin(phi) + r_dphi_dt * Math.cos(phi);
      const v = Math.hypot(vx, vy);
      const theta = Math.atan2(vy, vx);
      return [x, y, v, theta];
    }
    // complete fallback
    return [this.x0, this.y0, 0.0, 0.0];
  }
}

/* ---------------- Simulation controller & UI ---------------- */
class Simulation {
  constructor(params={}) {
    this.initial_T_total = params.T_total ?? DEFAULT_TTOTAL;
    this.initial_nframes = params.nframes ?? DEFAULT_NFRAMES;
    this.T_total = Number(this.initial_T_total);
    this.nframes = Number(this.initial_nframes);
    this.t_common = linspace(0.0, this.T_total, this.nframes);
    this.dt = (this.t_common.length>1) ? (this.t_common[1] - this.t_common[0]) : 1.0;
    this.r_escape = params.r_escape ?? DEFAULT_R_ESCAPE;

    this.particle_counter = 0;
    this.particles = [];
    this.current_frame = 0;
    this.show_trails = true;
    this.playing = true;
    this.highlighted = new Set();

    // canvas & drawing
    this.canvas = document.getElementById("simCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.W = this.canvas.width;
    this.H = this.canvas.height;
    this.R_display = 80.0;
    this.autoZoom = true; 
    this.scale = Math.min(this.W, this.H) / (2*this.R_display); // pixels per unit
    this.origin = [this.W/2, this.H/2];

    // UI elements
    this.dom = {
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
      zoomSlider: document.getElementById("zoomSlider"), // ズームスライダー
      zoomVal: document.getElementById("zoomVal")        // ズーム表示値
    };

    this._bindUI();
    this._updatePreview();
    this._tickHandle = null;
    this._startAnimation();
  }

  _bindUI() {
    this.dom.sliderX.oninput = () => this._updatePreview();
    this.dom.sliderY.oninput = () => this._updatePreview();
    this.dom.sliderTheta.oninput = () => this._updatePreview();
    this.dom.sliderV.oninput = () => this._updatePreview();

    this.dom.btnAdd.onclick = () => this._onAddParticle();
    this.dom.btnReset.onclick = () => this._onReset();
    this.dom.btnTrails.onclick = () => this._toggleTrails();
    this.dom.btnPlay.onclick = () => this._togglePlay();


    //  ズームスライダーイベント
    if (this.dom.zoomSlider) {
      this.dom.zoomSlider.addEventListener("input", () => {
        this.autoZoom = false; // オートズームOFF
        const z = Number(this.dom.zoomSlider.value);
        this.setZoom(z);
        if (this.dom.zoomVal) this.dom.zoomVal.textContent = z;
      });
    }

    // スライダーを上下キーで微調整できるようにする
    const sliders = [this.dom.sliderX, this.dom.sliderY, this.dom.sliderTheta, this.dom.sliderV];
    sliders.forEach(slider => {
      slider.addEventListener("keydown", (ev) => {
        let step = parseFloat(slider.step) || 0.1; // HTMLでstepを指定、なければ0.1
        let value = parseFloat(slider.value);

        if (ev.key === "ArrowUp") {
          slider.value = Math.min(value + step, slider.max);
          slider.dispatchEvent(new Event("input"));
          ev.preventDefault();
        }
        if (ev.key === "ArrowDown") {
          slider.value = Math.max(value - step, slider.min);
          slider.dispatchEvent(new Event("input"));
          ev.preventDefault();
        }
      });
    });

  // --- 各スライダーの上下ボタン処理（長押し対応版: PC+スマホ） ---
  document.querySelectorAll(".btn-up, .btn-down").forEach(btn => {
    let holdTimer;   // setInterval の ID 保管用

    const stepSlider = () => {
      const slider = document.getElementById(btn.dataset.target);
      if (!slider) return;

      const step = Number(slider.step) || 1;
      let val = Number(slider.value);

      if (btn.classList.contains("btn-up")) {
        val += step;
      } else {
        val -= step;
      }

      // 範囲を超えないようクリップ
      val = Math.max(Number(slider.min), Math.min(Number(slider.max), val));

      slider.value = val;
      slider.dispatchEvent(new Event("input")); // 値表示とプレビュー更新
    };

    // クリック（PCで1回だけ）
    btn.addEventListener("click", stepSlider);

    // --- マウス操作 ---
    btn.addEventListener("mousedown", () => {
      stepSlider(); // 押した瞬間も1回
      holdTimer = setInterval(stepSlider, 200);
    });
    ["mouseup", "mouseleave"].forEach(ev =>
      btn.addEventListener(ev, () => clearInterval(holdTimer))
    );

    // --- タッチ操作（スマホ/タブレット） ---
    btn.addEventListener("touchstart", (e) => {
      e.preventDefault(); // 画面スクロール抑制
      stepSlider();
      holdTimer = setInterval(stepSlider, 200);
    });
    ["touchend", "touchcancel"].forEach(ev =>
      btn.addEventListener(ev, () => clearInterval(holdTimer))
    );
  });

  }

    /* ---------------- ズーム制御 ---------------- */
  setZoom(R_display) {
    this.R_display = R_display;
    this.scale = Math.min(this.W, this.H) / (2*this.R_display);
  }

  worldToCanvas(x,y) {
    // world coords centered at (0,0) -> canvas px
    this.scale = Math.min(this.W, this.H) / (2*this.R_display);
    return [this.origin[0] + x*this.scale, this.origin[1] - y*this.scale];
  }

  _updatePreview() {
    const x = Number(this.dom.sliderX.value);
    const y = Number(this.dom.sliderY.value);
    const theta = Number(this.dom.sliderTheta.value);
    const v = Number(this.dom.sliderV.value);
    this.dom.valX.textContent = x.toFixed(2);
    this.dom.valY.textContent = y.toFixed(2);
    this.dom.valTheta.textContent = theta.toFixed(2);
    this.dom.valV.textContent = v.toFixed(3);
    this._drawScene(); // re-render to show preview
  }

  _onAddParticle() {
    const x = Number(this.dom.sliderX.value);
    const y = Number(this.dom.sliderY.value);
    const theta = Number(this.dom.sliderTheta.value);
    const v = Number(this.dom.sliderV.value);
    const vx = v * Math.cos(theta);
    const vy = v * Math.sin(theta);
    const idx = Math.min(this.current_frame, this.t_common.length-1);
    const t_inj = this.t_common[idx];
    const color = `hsl(${(this.particle_counter*37)%360} 80% 45%)`;
    const name = `P${String(this.particle_counter+1).padStart(3,'0')}`;
    try {
      const p = new Particle(x,y,vx,vy,t_inj, {name, color, r_escape: this.r_escape, atol: DEFAULT_ATOL, rtol: DEFAULT_RTOL});
      p.resample_observer_time(this.t_common);
      this.particles.push(p);
      this.particle_counter++;
      this._updateTable();
      console.log(`[add_particle] ${name} at (${x.toFixed(3)},${y.toFixed(3)}) v=(${vx.toFixed(3)},${vy.toFixed(3)}) t_inj=${t_inj.toFixed(3)}`);
    } catch (e) {
      console.warn("[skip particle]", e);
    }
  }

  _onReset() {
    // clear arrays and reset
    this.particles = [];
    this.particle_counter = 0;
    this.current_frame = 0;
    this.R_display = 80.0;
    this.highlighted.clear();
    this.t_common = linspace(0.0, this.initial_T_total, this.initial_nframes);
    this.dt = (this.t_common.length>1) ? (this.t_common[1]-this.t_common[0]) : 1.0;
    this._updatePreview();
    this._updateTable();
    this.dom.status.textContent = "Status: reset";
    console.log("[reset] Simulation fully reset");
  }

  _toggleTrails() {
    this.show_trails = !this.show_trails;
    this.dom.status.textContent = `Trails: ${this.show_trails ? "ON" : "OFF"}`;
  }

  _togglePlay() {
    if (this.playing) {
      this.playing = false;
      if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
      this.dom.btnPlay.textContent = "Play";
    } else {
      this.playing = true;
      this._startAnimation();
      this.dom.btnPlay.textContent = "Pause";
    }
  }

  _startAnimation() {
    if (this._tickHandle) clearInterval(this._tickHandle);
    this._tickHandle = setInterval(()=>this._updateFrame(), 30); // 30 ms ~ Python
    this.playing = true;
    this.dom.btnPlay.textContent = "Pause";
  }

  _extend_t_common(n_extra_frames=null) {
    if (n_extra_frames === null) n_extra_frames = this.initial_nframes;
    const last_t = this.t_common[this.t_common.length-1];
    const new_times = new Array(n_extra_frames);
    for (let i=0;i<n_extra_frames;i++) new_times[i] = last_t + this.dt*(i+1);
    this.t_common = this.t_common.concat(new_times);
    for (const p of this.particles) {
      const required_index = this.t_common.length - 1;
      p.ensure_tau_covers(this.t_common, required_index);
      p.resample_observer_time(this.t_common);
    }
  }

  _state_for(p, t_now, idx) {
    if (t_now < p.t_inject - 1e-12) return "pending";
    if (!p.alive) return p.state;
    if (!p.x_obs || p.x_obs.length === 0) return "integrating";
    const upto = Math.min(idx, p.x_obs.length-1);
    let valid = false;
    for (let i=0;i<=upto;i++) if (Number.isFinite(p.x_obs[i]) && Number.isFinite(p.y_obs[i])) { valid = true; break; }
    if (valid) return "active";
    const t_rel_needed = t_now - p.t_inject;
    if (p.t_tau.length>0 && t_rel_needed > p.t_tau[p.t_tau.length-1]) return "integrating";
    return "inactive";
  }

  _updateTable() {
  // 初回だけ空のオブジェクトを用意
  if (!this.preserve) {
    this.preserve = {};
  }

  const tbody = this.dom.tableBody;

  // 既存テーブルからチェック状態を保存
  for (const tr of tbody.children) {
    const name = tr.dataset.name;
    if (!name) continue;
    const chk = tr.querySelector('td.checkbox-cell input[type=checkbox]');
    if (chk) {
      this.preserve[name] = chk.checked;
    }
  }

  // テーブルをクリア
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

  // 粒子が存在しない場合の空行処理
  if (this.particles.length === 0) {
    const tr = document.createElement("tr");
    for (let i = 0; i < 9; i++) {
      const td = document.createElement("td");
      if (i === 1) td.textContent = "";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
    return;
  }

  const idx = Math.min(this.current_frame, Math.max(0, this.t_common.length - 1));
  const t_now = this.t_common[idx];

  for (const p of this.particles) {
    let x = "", y = "", v = "", theta = "", E = "", L = "", state = "";
    try {
      // 追加：r_escape により統合が停止（逃逸）した粒子は table 上で x,y を Infinity とする
      if (!p.alive && p.state === "escaping") {
        x = "--";
        y = "--";
        v = "--";
        theta = "--";
      } else {
        const rr = p.current_xyv(t_now, idx);
        x = Number(rr[0]).toPrecision(6);
        y = Number(rr[1]).toPrecision(6);
        v = Number(rr[2]).toPrecision(6);
        theta = Number(rr[3]).toFixed(6);
      }
    } catch (e) {
      x = p.x0.toPrecision(6);
      y = p.y0.toPrecision(6);
      v = "0";
      theta = "0.000";
    }
    state = this._state_for(p, t_now, idx);
    E = Number(p.E).toPrecision(6);
    L = Number(p.L).toPrecision(6);

    const tr = document.createElement("tr");
    tr.dataset.name = p.name;

// --- Select 列（チェックボックス） ---
    const tdSel = document.createElement("td");
    tdSel.className = "checkbox-cell";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = this.preserve[p.name] === true;  // 状態復元

    input.onchange = () => {
      this.preserve[p.name] = input.checked;  // 状態保存
      if (input.checked) {
        this.highlighted.add(p.name);
      } else {
        this.highlighted.delete(p.name);
      }
      this._drawScene();  // 再描画
    };

    tdSel.appendChild(input);
    tr.appendChild(tdSel);

    // --- 他のセル ---
    const cells = [p.name, x, y, theta, v, E, L, state];
    for (let c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }

    tbody.appendChild(tr);
  }
}


  /* ---------------- draw functions ---------------- */
  _clearCanvas() {
    this.ctx.clearRect(0,0,this.W,this.H);
  }

  _drawBackground() {
    // dark background already set by css; draw central black circle (Rg)
    const [cx, cy] = this.worldToCanvas(0,0);
    const pxR = Math.max(1, Math.abs(Rg) * this.scale);
    // subtle halo
    this.ctx.beginPath();
    this.ctx.fillStyle = "#000";
    this.ctx.arc(cx, cy, pxR, 0, Math.PI*2);
    this.ctx.fill();
    // rim
    this.ctx.lineWidth = 1;
    this.ctx.strokeStyle = "#555";
    this.ctx.stroke();
  }

  _drawTrailsAndPoints() {
    // draw each particle trail and point
    const frame = this.current_frame;
    for (let i=0;i<this.particles.length;i++) {
      const p = this.particles[i];
      const color = p.color || `hsl(${(i*37)%360} 80% 45%)`;
      // trail
      if (this.show_trails && p.x_obs) {
        this.ctx.beginPath();
        let started=false;
        for (let j=0;j<=Math.min(frame, p.x_obs.length-1); j++) {
          const xx = p.x_obs[j], yy = p.y_obs[j];
          if (!Number.isFinite(xx) || !Number.isFinite(yy)) { started=false; continue; }
          const [px, py] = this.worldToCanvas(xx, yy);
          if (!started) { this.ctx.moveTo(px, py); started=true; }
          else this.ctx.lineTo(px, py);
        }
        this.ctx.lineWidth = this.highlighted.has(p.name) ? 6.5 : 2.5;
        this.ctx.strokeStyle = this.highlighted.has(p.name) ? "gold" : color;
        this.ctx.globalAlpha = this.highlighted.has(p.name) ? 1.0 : 0.5;
        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;
      }
      // current point
      let xnow = (p.x_obs && frame < p.x_obs.length) ? p.x_obs[frame] : NaN;
      let ynow = (p.y_obs && frame < p.y_obs.length) ? p.y_obs[frame] : NaN;
      if (!Number.isFinite(xnow) || !Number.isFinite(ynow)) {
        // draw nothing
      } else {
        const [px,py] = this.worldToCanvas(xnow, ynow);
        const ms = this.highlighted.has(p.name) ? 12 : 6.5;
        this.ctx.globalAlpha = this.highlighted.has(p.name) ? 1.0 : 0.8;
        this.ctx.beginPath();
        this.ctx.fillStyle = this.highlighted.has(p.name) ? "gold" : color;
        this.ctx.arc(px, py, ms, 0, Math.PI*2);
        this.ctx.fill();
        // outline
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = "#222";
        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;

        //  オートズーム処理はフラグで制御
        if (this.autoZoom &&
            (Math.abs(xnow) > 0.95 * this.R_display || Math.abs(ynow) > 0.95 * this.R_display)) {
          this.R_display *= 1.5;
          this.scale = Math.min(this.W, this.H) / (2*this.R_display);
          if (this.dom.zoomSlider) {
            this.dom.zoomSlider.value = this.R_display;
            if (this.dom.zoomVal) this.dom.zoomVal.textContent = this.R_display;
          }
        }
      }
    }
  }

  _drawPreview() {
    const x = Number(this.dom.sliderX.value);
    const y = Number(this.dom.sliderY.value);
    const theta = Number(this.dom.sliderTheta.value);
    const v = Number(this.dom.sliderV.value);
    const [px, py] = this.worldToCanvas(x,y);
    // point
    this.ctx.beginPath();
    this.ctx.fillStyle = "red";
    this.ctx.arc(px, py, 6.5, 0, Math.PI*2);
    this.ctx.fill();
    // arrow
    const arrowLen = (v === 0 ? 0 : (v * this.R_display/8 + this.R_display/18));
    if (arrowLen > 0) {
      const dx = arrowLen * Math.cos(theta);
      const dy = arrowLen * Math.sin(theta);
      const [ex, ey] = this.worldToCanvas(x+dx, y+dy);
      this._drawArrow(px, py, ex, ey, "red");
    }
  }

  _drawArrow(sx, sy, ex, ey, color) {
    const ctx = this.ctx;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // arrow head
    const ang = Math.atan2(ey - sy, ex - sx);
    const size = 20;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - size*Math.cos(ang - Math.PI/6), ey - size*Math.sin(ang - Math.PI/6));
    ctx.lineTo(ex - size*Math.cos(ang + Math.PI/6), ey - size*Math.sin(ang + Math.PI/6));
    ctx.closePath();
    ctx.fill();
  }

  _drawScene() {
    this._clearCanvas();
    // grid (optional)
    // draw central circle
    this._drawBackground();
    // trails and points
    this._drawTrailsAndPoints();
    // preview overlay
    this._drawPreview();
  }

  _updateFrame() {
    // frame progression
    this.current_frame++;
    if (this.current_frame >= this.t_common.length - 10) {
      this._extend_t_common(null);
    }
    const t_now = this.t_common[Math.min(this.current_frame, this.t_common.length-1)];
    // ensure particle data coverage
    for (const p of this.particles) {
      try {
        if (p.t_tau.length === 0 || (t_now - p.t_inject) > (p.t_tau.length>0 ? p.t_tau[p.t_tau.length-1] : -Infinity)) {
          p.ensure_tau_covers(this.t_common, Math.min(this.current_frame, this.t_common.length-1));
          p.resample_observer_time(this.t_common);
        }
      } catch (e) { console.warn("resample failed",e); }
    }
    // redraw
    this._drawScene();
    // update table
    this._updateTable();

    const ot = document.getElementById("observerTime");
        if (ot) {
            ot.textContent = "Observer Time: " + t_now.toFixed(6);
        }

  }
}

/* ---------------- instantiate simulation ---------------- */
const sim = new Simulation({T_total: DEFAULT_TTOTAL, nframes: DEFAULT_NFRAMES});
sim.dom.status.textContent = "Status: running";

/* ---------------- Wire canvas scaling to window size ---------------- */

(function fitCanvas(){
  const canvas = sim.canvas;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // 正方形（幅と高さの小さい方に合わせる）
    const size = Math.min(rect.width, rect.height);

    // 見た目サイズ（CSS px）
    canvas.style.width  = size + "px";
    canvas.style.height = size + "px";

    // 内部解像度（物理解像度）を dpr 倍にする
    canvas.width  = Math.round(size * dpr); 
    canvas.height = Math.round(size * dpr);

    // シミュレーション用に幅・高さを更新
    sim.W = canvas.width;
    sim.H = canvas.height;
    sim.origin = [sim.W/2, sim.H/2];


    sim._drawScene()
  }

  // 初期化
  resize();

  // ウィンドウリサイズ時
  window.addEventListener("resize", () => { setTimeout(resize, 50); });

  let lastDPR = window.devicePixelRatio;
  setInterval(() => {
    if (window.devicePixelRatio !== lastDPR) {
      lastDPR = window.devicePixelRatio;
      resize();
    }
  }, 500);
})();


