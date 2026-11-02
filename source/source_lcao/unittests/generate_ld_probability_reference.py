#!/usr/bin/env python3
"""Independent Decimal reference for SHARC Eq.12; no ABACUS imports.

Constructs exact Householder matrices and complex phases, then evaluates the
published diagonal-denominator formula at 80/120 digits. The production code
instead sums off-diagonal weights. No probability renormalization is used.
Usage: python3 generate_ld_probability_reference.py [output_path]
"""

from decimal import Decimal as D, localcontext
from pathlib import Path
import sys


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conj(a):
    return (a[0], -a[1])


def norm(a):
    return mul(a, conj(a))[0]


def output_number(value):
    # Round the reference to 60 decimal places before double serialization;
    # this removes only residuals of exact cancellations at working precision.
    if abs(value) < D('1e-60'):
        return '0'
    return f'{float(value):.17g}'


def build(precision):
    with localcontext() as context:
        context.prec = precision
        zero = (D(0), D(0))
        phases = [(D(1), D(0)), (D(0), D(1)), (D(-1), D(0)), (D(0), D(-1))]
        definitions = [
            ([1, 1, 1], [(1, 0), (0, 0), (0, 0)], 0, 0),
            ([1, 1, 1], [(1, 0), (2, 0), (-2, 0)], 0, 0),
            ([1, 1, 1], [(1, 0), (0, 1), (1, 0)], 0, 0),
            ([1, 1, 1], [(1, 1), (-2, 1), (0, 1)], 1, 0),
            ([1, 2, 3], [(1, 0), (0, 2), (-2, 0)], 2, 1),
            ([1, 2, 3], [(1, 1), (2, -1), (-1, 2)], 1, 2),
            ([1, 2, 3, 4], [(1, 0), (0, 2), (-1, 0), (1, 0)], 1, 1),
            ([1, 2, 3, 4], [(0, 0), (0, 0), (1, 0), (0, 0)], 2, 2),
        ]
        lines = [str(len(definitions))]
        for vector, amplitudes, active, phase_shift in definitions:
            n = len(vector)
            v = [D(x) for x in vector]
            vnorm = sum(x * x for x in v)
            matrix = []
            for i in range(n):
                row = []
                for j in range(n):
                    entry = (D(i == j) - 2 * v[i] * v[j] / vnorm, D(0))
                    if phase_shift:
                        entry = mul(mul(phases[(i + phase_shift) % 4], entry), phases[j % 4])
                    row.append(entry)
                matrix.append(row)
            a = [(D(x), D(y)) for x, y in amplitudes]
            length = sum(norm(x) for x in a).sqrt()
            a = [(x / length, y / length) for x, y in a]
            b = []
            for row in matrix:
                value = zero
                for x, y in zip(row, a):
                    value = add(value, mul(x, y))
                b.append(value)
            p = norm(a[active])
            pnew = norm(b[active])
            q = [D(0)] * n
            if pnew < p:
                denominator = p - mul(mul(b[active], conj(matrix[active][active])), conj(a[active]))[0]
                for j in range(n):
                    if j != active:
                        z = mul(mul(b[j], conj(matrix[j][active])), conj(a[active]))[0]
                        q[j] = max(D(0), (1 - pnew / p) * z / denominator)
            reject = sum(q) > D(1) + D('1e-12')
            lines.append(f'{n} {active} {int(reject)}')
            for row in matrix:
                lines.extend(f'{output_number(x)} {output_number(y)}' for x, y in row)
            for values in (a, b):
                lines.extend(f'{output_number(x)} {output_number(y)}' for x, y in values)
            lines.append(' '.join(output_number(x) for x in q))
        return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    result = build(80)
    if result != build(120):
        raise RuntimeError('Reference is not stable between 80 and 120 digits')
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('ld_probability_reference.txt')
    target.write_text(result, encoding='utf-8')
    print(f'Wrote 8 independent references; 80/120-digit agreement: {target}')
