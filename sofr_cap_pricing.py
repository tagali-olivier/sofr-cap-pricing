"""
Calibrage d'une courbe de swap SOFR et pricing d'un cap 5 ans à 3 %
====================================================================

Projet Quant FICC.
    1. Bootstrap d'une courbe de taux zéro (capitalisation continue) à partir
       des taux de swap SOFR 1 an, 3 ans, 5 ans.
    2. Calcul des taux forwards trimestriels sur la courbe calibrée.
    3. Pricing d'un cap (et du floor associé) par la formule de Black, caplet par caplet.
    4. Contrôles : re-pricing des swaps, parité cap-floor, cas de la courbe plate.
    5. Sensibilités : prix vs volatilité, vega, choc parallèle, deltas par pilier.

Conventions retenues (voir le rapport, section « Données et conventions ») :
    - Base Act/360 pour les fractions d'année (accruals ET axe des temps).
    - Taux zéro r(t) en capitalisation continue : P(0, t) = exp(-r(t) t).
    - Jambe fixe des swaps annuelle (cf. l'équation de bootstrap de l'énoncé).
    - Cap trimestriel, 20 périodes ; le 1er caplet (fixé aujourd'hui) n'a pas
      d'optionalité, il est exclu du prix (sa valeur intrinsèque est nulle ici).

Exécution :  python sofr_cap_pricing.py      (génère les graphiques dans ./figures)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")  # backend sans écran : on enregistre les figures en PNG
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Paramètres du projet
# ---------------------------------------------------------------------------
VALUATION_DATE = date(2026, 11, 3)          # date de valorisation
SWAP_TENORS = np.array([1, 3, 5])            # maturités cotées (années)
SWAP_RATES = np.array([0.020, 0.025, 0.030]) # taux de swap SOFR cotés

NOTIONAL = 10_000_000                        # N = 10 M$
STRIKE = 0.03                                # K = 3 %
CAP_MATURITY_YEARS = 5
CAP_FREQ_MONTHS = 3                          # trimestriel
SIGMA = 0.20                                 # vol de Black (lognormale) 20 %
DAY_COUNT = 360.0                            # Act/360


def yearfrac(d1: date, d2: date) -> float:
    """Fraction d'année Act/360 entre deux dates."""
    return (d2 - d1).days / DAY_COUNT


def schedule(start: date, years: int, months_step: int) -> list[date]:
    """Échéancier de dates (sans ajustement jours ouvrés), start inclus."""
    n = years * 12 // months_step
    return [start + relativedelta(months=months_step * k) for k in range(n + 1)]


# ---------------------------------------------------------------------------
# 1. Courbe de taux : objet + bootstrap
# ---------------------------------------------------------------------------
@dataclass
class ZeroCurve:
    """Courbe de taux zéro continus, interpolée entre les piliers bootstrappés.

    method = "linear"  : interpolation linéaire des taux zéro (robuste, forwards en escalier)
    method = "cubic"   : spline cubique naturelle sur les taux zéro (courbe lisse)
    Extrapolation plate avant le premier pilier et après le dernier.
    """

    times: np.ndarray   # piliers en années Act/360
    zeros: np.ndarray   # taux zéro continus aux piliers
    method: str = "cubic"

    def __post_init__(self):
        if self.method == "cubic":
            self._spline = CubicSpline(self.times, self.zeros, bc_type="natural")

    def r(self, t):
        t = np.clip(np.asarray(t, dtype=float), self.times[0], self.times[-1])
        if self.method == "cubic":
            return self._spline(t)
        return np.interp(t, self.times, self.zeros)

    def df(self, t):
        """Facteur d'actualisation P(0,t) = exp(-r(t) t)."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.r(t) * t)

    def forward(self, t1, t2):
        """Taux forward simple (linéaire) entre t1 et t2 :
        L(t1,t2) = (P(t1)/P(t2) - 1) / tau  — identique à la formule de l'énoncé
        (e^{r2 t2} - e^{r1 t1}) / (e^{r1 t1} tau)."""
        tau = np.asarray(t2) - np.asarray(t1)
        return (self.df(t1) / self.df(t2) - 1.0) / tau


def bootstrap_swap_curve(T, swap_rates, valuation_date=VALUATION_DATE,
                         method="cubic") -> tuple[ZeroCurve, dict]:
    """Bootstrap des taux zéro à partir de swaps à jambe fixe annuelle.

    Équation d'un swap au pair (énoncé) :
        1 = sum_{i=1}^{n} s(T) tau_i P(0,t_i) + P(0,T)
    Avec des coupons annuels et un pilier par année, chaque nouvelle maturité
    n'ajoute qu'une seule inconnue P(0,T_n) ; on l'isole directement :
        P(0,T_n) = (1 - s_n * sum_{i<n} tau_i P(0,t_i)) / (1 + s_n tau_n)

    Les maturités non cotées (2 et 4 ans) sont obtenues par interpolation
    linéaire des taux de swap (hypothèse standard de « par rate interpolation »).
    """
    T = np.asarray(T, dtype=int)
    swap_rates = np.asarray(swap_rates, dtype=float)

    full_T = np.arange(1, T.max() + 1)                       # 1, 2, 3, 4, 5
    full_s = np.interp(full_T, T, swap_rates)                # 2 %, 2.25 %, 2.5 %, 2.75 %, 3 %

    dates = schedule(valuation_date, int(T.max()), 12)       # dates annuelles
    t = np.array([yearfrac(valuation_date, d) for d in dates[1:]])
    tau = np.diff(np.concatenate([[0.0], t]))                # accruals Act/360 (~1.0139)

    dfs = np.empty(len(full_T))
    for n in range(len(full_T)):
        annuity_known = np.sum(tau[:n] * dfs[:n])            # coupons déjà actualisés
        dfs[n] = (1.0 - full_s[n] * annuity_known) / (1.0 + full_s[n] * tau[n])

    zeros = -np.log(dfs) / t                                 # r(t) continu
    curve = ZeroCurve(t, zeros, method=method)
    info = {"T": full_T, "s": full_s, "t": t, "tau": tau, "df": dfs, "zeros": zeros}
    return curve, info


def swap_par_rate(curve: ZeroCurve, years: int, valuation_date=VALUATION_DATE) -> float:
    """Taux de swap au pair recalculé sur la courbe (jambe fixe annuelle).
    Sert de test d'ajustement : il doit retrouver la cotation d'entrée."""
    dates = schedule(valuation_date, years, 12)
    t = np.array([yearfrac(valuation_date, d) for d in dates[1:]])
    tau = np.diff(np.concatenate([[0.0], t]))
    annuity = np.sum(tau * curve.df(t))
    return (1.0 - curve.df(t[-1])) / annuity


# ---------------------------------------------------------------------------
# 2. Pricing Black des caplets / floorlets
# ---------------------------------------------------------------------------
def black_caplet(F, K, sigma, t_fix, tau, df_pay, notional, is_cap=True):
    """Formule de Black (forward lognormal) :
        Caplet   = N tau P(0,t_pay) [F N(d1) - K N(d2)]
        Floorlet = N tau P(0,t_pay) [K N(-d2) - F N(-d1)]
    d1 = [ln(F/K) + sigma^2 t_fix / 2] / (sigma sqrt(t_fix)), d2 = d1 - sigma sqrt(t_fix)
    La variance s'accumule jusqu'à la date de FIXING t_{i-1}, pas jusqu'au paiement.
    """
    F, t_fix = np.asarray(F, float), np.asarray(t_fix, float)
    sd = sigma * np.sqrt(t_fix)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    if is_cap:
        payoff = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        payoff = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return notional * tau * df_pay * payoff


def caplet_table(curve: ZeroCurve, K=STRIKE, sigma=SIGMA, notional=NOTIONAL,
                 years=CAP_MATURITY_YEARS, months=CAP_FREQ_MONTHS,
                 valuation_date=VALUATION_DATE, is_cap=True) -> dict:
    """Construit l'échéancier du cap et valorise chaque caplet.

    Période i : fixing en t_{i-1}, paiement en t_i, accrual tau_i (Act/360).
    La période 1 (fixing à t=0) est déjà connue : pas d'optionalité, on l'exclut
    du cap et on reporte seulement sa valeur intrinsèque.
    """
    dates = schedule(valuation_date, years, months)
    t = np.array([yearfrac(valuation_date, d) for d in dates])  # t_0 = 0, ..., t_20
    t_fix, t_pay = t[:-1], t[1:]
    tau = t_pay - t_fix

    F = curve.forward(t_fix, t_pay)       # forwards SOFR trimestriels (proxy du taux composé)
    df_pay = curve.df(t_pay)

    values = np.zeros_like(F)
    # Périodes 2..20 : vraie option (fixing futur)
    values[1:] = black_caplet(F[1:], K, sigma, t_fix[1:], tau[1:], df_pay[1:],
                              notional, is_cap)
    # Période 1 : flux déterministe (intrinsèque), non inclus dans le prix du cap
    first_intrinsic = notional * tau[0] * df_pay[0] * max((F[0] - K) if is_cap else (K - F[0]), 0.0)

    return {"t_fix": t_fix, "t_pay": t_pay, "tau": tau, "F": F, "df": df_pay,
            "values": values, "price": values[1:].sum(),
            "first_intrinsic": first_intrinsic, "pay_dates": dates[1:]}


def cap_price(N, K, curve, sigma, is_cap=True) -> float:
    """Signature demandée par l'énoncé : prix du cap = somme des caplets."""
    return caplet_table(curve, K=K, sigma=sigma, notional=N, is_cap=is_cap)["price"]


# ---------------------------------------------------------------------------
# 3. Sensibilités
# ---------------------------------------------------------------------------
class ShiftedCurve(ZeroCurve):
    """Courbe choquée parallèlement sur les taux zéro : r(t) + shift."""

    def __init__(self, base: ZeroCurve, shift: float):
        self.base, self.shift = base, shift
        self.times, self.zeros, self.method = base.times, base.zeros + shift, base.method

    def r(self, t):
        return self.base.r(t) + self.shift


def sensitivities(curve: ZeroCurve, base_price: float) -> dict:
    """Vega, choc parallèle et deltas par pilier de swap (key-rate)."""
    out = {}
    # Vega : +1 point de vol (20 % -> 21 %), différence centrée
    up = cap_price(NOTIONAL, STRIKE, curve, SIGMA + 0.01)
    dn = cap_price(NOTIONAL, STRIKE, curve, SIGMA - 0.01)
    out["vega_1pt"] = (up - dn) / 2

    # Choc parallèle de la courbe zéro : +/-1 pb et +/-100 pb
    for bp in (1, 100):
        h = bp * 1e-4
        pu = cap_price(NOTIONAL, STRIKE, ShiftedCurve(curve, +h), SIGMA)
        pd = cap_price(NOTIONAL, STRIKE, ShiftedCurve(curve, -h), SIGMA)
        out[f"parallel_+{bp}bp"] = pu - base_price
        out[f"parallel_-{bp}bp"] = pd - base_price
    out["dv01"] = (out["parallel_+1bp"] - out["parallel_-1bp"]) / 2

    # Deltas par pilier : on choque UNE cotation de swap de 1 pb et on re-bootstrappe
    krd = {}
    for j, T in enumerate(SWAP_TENORS):
        bumped = SWAP_RATES.copy()
        bumped[j] += 1e-4
        c_up, _ = bootstrap_swap_curve(SWAP_TENORS, bumped, method=curve.method)
        krd[f"{T}Y"] = cap_price(NOTIONAL, STRIKE, c_up, SIGMA) - base_price
    out["key_rate_delta_1bp"] = krd
    return out


def cap_floor_parity(curve: ZeroCurve, table: dict) -> tuple[float, float]:
    """Cap - Floor = swap payeur forward-start (payer K, recevoir SOFR), périodes 2..20.
    Valeur du swap = N sum tau_i P(0,t_i) (F_i - K) — indépendante du modèle de vol."""
    floor = cap_price(NOTIONAL, STRIKE, curve, SIGMA, is_cap=False)
    s = slice(1, None)
    swap = NOTIONAL * np.sum(table["tau"][s] * table["df"][s] * (table["F"][s] - STRIKE))
    return table["price"] - floor, swap


# ---------------------------------------------------------------------------
# 4. Graphiques
# ---------------------------------------------------------------------------
def make_figures(curve: ZeroCurve, info: dict, table: dict, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    tt = np.linspace(0.0, info["t"][-1], 300)   # extrapolation plate avant 1 an

    # (a) Courbe calibrée : swaps cotés, taux zéro, forwards 3M
    lin_curve, _ = bootstrap_swap_curve(SWAP_TENORS, SWAP_RATES, method="linear")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(SWAP_TENORS, SWAP_RATES * 100, "o", ms=8, color="black", label="Swaps SOFR cotés")
    ax.plot(tt, curve.r(tt) * 100, lw=2, label="Taux zéro r(t) — spline cubique")
    ax.plot(tt, lin_curve.r(tt) * 100, "--", lw=1, label="Taux zéro r(t) — linéaire")
    ax.step(np.append(table["t_fix"], table["t_pay"][-1]), np.append(table["F"], table["F"][-1]) * 100,
            where="post", lw=1.5,
            label="Forwards 3M L(t_{i-1}, t_i)")
    ax.axhline(STRIKE * 100, color="red", ls=":", label="Strike du cap 3 %")
    ax.set(title="Courbe SOFR calibrée", xlabel="Maturité (années, Act/360)", ylabel="Taux (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/1_courbe_calibree.png", dpi=150); plt.close(fig)

    # (b) Valeur des caplets
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(table["t_pay"][1:], table["values"][1:] / 1e3, width=0.18)
    ax.set(title="Valeur de chaque caplet (σ = 20 %)", xlabel="Date de paiement (années)",
           ylabel="Valeur (k$)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{outdir}/2_caplets.png", dpi=150); plt.close(fig)

    # (c) Prix du cap vs volatilité
    sigmas = np.linspace(0.05, 0.60, 56)
    prices = [cap_price(NOTIONAL, STRIKE, curve, s) for s in sigmas]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(sigmas * 100, np.array(prices) / 1e3, lw=2)
    ax.axvline(SIGMA * 100, color="red", ls=":", label="σ de référence 20 %")
    ax.set(title="Prix du cap 5 ans à 3 % en fonction de σ", xlabel="Volatilité de Black σ (%)",
           ylabel="Prix du cap (k$)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/3_cap_vs_sigma.png", dpi=150); plt.close(fig)

    # (d) Prix du cap vs choc parallèle de taux
    shifts = np.linspace(-0.015, 0.015, 31)
    p_shift = [cap_price(NOTIONAL, STRIKE, ShiftedCurve(curve, h), SIGMA) for h in shifts]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(shifts * 1e4, np.array(p_shift) / 1e3, lw=2)
    ax.set(title="Prix du cap en fonction d'un choc parallèle de la courbe",
           xlabel="Choc parallèle (pb)", ylabel="Prix du cap (k$)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/4_cap_vs_taux.png", dpi=150); plt.close(fig)
    return sigmas, prices


# ---------------------------------------------------------------------------
# 5. Programme principal
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()

    # --- Courbe ---
    curve, info = bootstrap_swap_curve(SWAP_TENORS, SWAP_RATES, method="cubic")
    print("=== Courbe bootstrappée (piliers) ===")
    print(f"{'T':>3} {'swap':>8} {'t Act/360':>10} {'P(0,T)':>10} {'r(T) cont.':>11}")
    for T, s, t, d, z in zip(info["T"], info["s"], info["t"], info["df"], info["zeros"]):
        print(f"{T:>3} {s*100:>7.3f}% {t:>10.4f} {d:>10.6f} {z*100:>10.4f}%")

    print("\n=== Test d'ajustement : re-pricing des swaps ===")
    for T, s in zip(info["T"], info["s"]):
        err = (swap_par_rate(curve, int(T)) - s) * 1e4
        print(f"{T}Y : coté {s*100:.4f}%  recalculé {swap_par_rate(curve, int(T))*100:.4f}%  "
              f"écart {err:+.2e} pb")

    # --- Cap ---
    table = caplet_table(curve)
    print("\n=== Caplets (σ = 20 %, K = 3 %) ===")
    print(f"{'#':>2} {'paiement':>11} {'t_fix':>6} {'tau':>6} {'Forward':>8} {'P(0,t)':>8} {'Valeur $':>11}")
    for i in range(len(table["F"])):
        print(f"{i+1:>2} {table['pay_dates'][i].isoformat():>11} {table['t_fix'][i]:>6.3f} "
              f"{table['tau'][i]:>6.4f} {table['F'][i]*100:>7.3f}% {table['df'][i]:>8.5f} "
              f"{table['values'][i]:>11,.0f}")
    price = table["price"]
    print(f"\nPrix du cap (caplets 2 à 20) : {price:,.0f} $  "
          f"= {price/NOTIONAL*1e4:.1f} pb du notionnel")
    print(f"Caplet 1 (fixé aujourd'hui, intrinsèque) : {table['first_intrinsic']:,.0f} $")

    # --- Contrôles ---
    cf, swap = cap_floor_parity(curve, table)
    floor = cap_price(NOTIONAL, STRIKE, curve, SIGMA, is_cap=False)
    print(f"\nFloor 5 ans à 3 % : {floor:,.0f} $")
    print(f"Parité : Cap - Floor = {cf:,.2f} $ | swap payeur forward = {swap:,.2f} $ "
          f"| écart {cf-swap:.2e} $")

    # --- Sensibilités ---
    sens = sensitivities(curve, price)
    print("\n=== Sensibilités ===")
    print(f"Vega (+1 pt de vol)        : {sens['vega_1pt']:,.0f} $")
    print(f"DV01 (+1 pb parallèle)     : {sens['dv01']:,.0f} $")
    print(f"Choc +100 pb / -100 pb     : {sens['parallel_+100bp']:+,.0f} $ / {sens['parallel_-100bp']:+,.0f} $")
    print("Deltas par cotation (+1 pb):", {k: round(v) for k, v in sens["key_rate_delta_1bp"].items()})

    # --- Cas particulier : courbe plate à 3 % ---
    flat_curve, flat_info = bootstrap_swap_curve(SWAP_TENORS, [0.03, 0.03, 0.03])
    flat_tab = caplet_table(flat_curve)
    print("\n=== Cas particulier : swaps plats à 3 % ===")
    print(f"Zéros : {np.round(flat_info['zeros']*100, 4)} %  (plats, comme attendu)")
    print(f"Forwards 3M : min {flat_tab['F'].min()*100:.4f}%  max {flat_tab['F'].max()*100:.4f}%")
    print(f"Cap : {flat_tab['price']:,.0f} $ | Floor : "
          f"{cap_price(NOTIONAL, STRIKE, flat_curve, SIGMA, False):,.0f} $")

    # --- Comparaison des interpolations ---
    lin_curve, _ = bootstrap_swap_curve(SWAP_TENORS, SWAP_RATES, method="linear")
    print(f"\nPrix du cap avec interpolation linéaire : {cap_price(NOTIONAL, STRIKE, lin_curve, SIGMA):,.0f} $")

    elapsed_calc = time.perf_counter() - t0
    make_figures(curve, info, table)
    print(f"\nTemps de calcul : {elapsed_calc*1000:.1f} ms "
          f"(total avec graphiques : {time.perf_counter()-t0:.2f} s)")


if __name__ == "__main__":
    main()
