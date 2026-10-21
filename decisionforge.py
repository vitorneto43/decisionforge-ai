"""DecisionForge AI MVP: auditable multi-perspective scenario analysis.

This is NOT a predictive AI model. Agents are deterministic lenses; Monte Carlo
samples assumptions supplied by the user. No external APIs or dependencies.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Assumptions:
    investment: float
    monthly_revenue: float
    monthly_cost: float
    months: int
    revenue_uncertainty: float
    cost_uncertainty: float
    monthly_growth: float = 0.0
    discount_rate_annual: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> 'Assumptions':
        a = cls(**data)
        if a.investment < 0 or a.monthly_revenue < 0 or a.monthly_cost < 0:
            raise ValueError('Valores monetários devem ser não negativos')
        if a.months < 1 or a.months > 600:
            raise ValueError('months deve estar entre 1 e 600')
        if not 0 <= a.revenue_uncertainty <= 1 or not 0 <= a.cost_uncertainty <= 1:
            raise ValueError('Incertezas devem estar entre 0 e 1')
        if not -0.99 < a.monthly_growth <= 1:
            raise ValueError('monthly_growth fora do intervalo aceito')
        if not 0 <= a.discount_rate_annual <= 1:
            raise ValueError('discount_rate_annual fora do intervalo aceito')
        return a


def simulate(a: Assumptions, n: int, seed: int) -> list[float]:
    if not 100 <= n <= 1_000_000:
        raise ValueError('simulations deve estar entre 100 e 1000000')
    rng = random.Random(seed)
    values = []
    discount = (1+a.discount_rate_annual)**(1/12)-1
    for _ in range(n):
        # Independent scenario-level shocks: an assumption, not a forecast.
        revenue_factor = rng.uniform(1-a.revenue_uncertainty, 1+a.revenue_uncertainty)
        cost_factor = rng.uniform(1-a.cost_uncertainty, 1+a.cost_uncertainty)
        npv = -a.investment
        for month in range(1, a.months+1):
            revenue = a.monthly_revenue * revenue_factor * (1+a.monthly_growth)**(month-1)
            cost = a.monthly_cost * cost_factor
            npv += (revenue-cost)/(1+discount)**month
        values.append(npv)
    values.sort()
    return values


def percentile(xs: list[float], p: float) -> float:
    pos = (len(xs)-1)*p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)


def perspectives(a: Assumptions, outcomes: list[float]) -> list[dict]:
    # Explicit checks, not LLM autonomous agents.
    checks = [
        ('finance', 'Fluxo de caixa', 'Resultado mensal base: %.2f' % (a.monthly_revenue-a.monthly_cost)),
        ('risk', 'Risco financeiro', 'Perda em %.1f%% dos cenários simulados' % (100*sum(v<0 for v in outcomes)/len(outcomes))),
        ('market', 'Sensibilidade da receita', 'Intervalo assumido de receita: ±%.1f%%' % (100*a.revenue_uncertainty)),
        ('operations', 'Sensibilidade de custos', 'Intervalo assumido de custos: ±%.1f%%' % (100*a.cost_uncertainty)),
        ('skeptic', 'Auditoria de premissas', 'Os cenários refletem hipóteses do usuário, não taxas de acerto validadas.'),
    ]
    return [{'agent': key, 'focus': name, 'finding': finding} for key, name, finding in checks]


def analyze(data: dict, simulations: int = 10000, seed: int = 42) -> dict:
    a = Assumptions.from_dict(data)
    outcomes = simulate(a, simulations, seed)
    losses = sum(value < 0 for value in outcomes)
    return {
        'model': 'illustrative_monte_carlo_v0.1',
        'warning': 'Não é uma previsão validada; resultados dependem das premissas.',
        'simulation_count': simulations,
        'seed': seed,
        'metrics': {
            'median_npv': round(percentile(outcomes, .5), 2),
            'p10_npv': round(percentile(outcomes, .1), 2),
            'p90_npv': round(percentile(outcomes, .9), 2),
            'fraction_negative_npv': round(losses/simulations, 4),
        },
        'agents': perspectives(a, outcomes),
        'assumptions': data,
    }


def main():
    parser = argparse.ArgumentParser(description='DecisionForge AI — protótipo auditável')
    parser.add_argument('scenario', type=Path, help='JSON de premissas')
    parser.add_argument('--simulations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.scenario.read_text(encoding='utf-8')), args.simulations, args.seed)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output+'\n', encoding='utf-8')
    else:
        print(output)

if __name__ == '__main__':
    main()
