from sympy import *
from sympy.core.mul import _keep_coeff
from sympy.printing.precedence import precedence
from sympy.printing.rust import RustCodePrinter
from sympy.codegen.ast import Assignment
import subprocess
import os
import numpy as np

coeff_a = IndexedBase("a", 6)
coeff_b = IndexedBase("b", 6)
x, y = symbols("x y", real=True)
p = Matrix([x, y])

def step(p: Matrix):
    x = p[0]
    y = p[1]
    return Matrix(
        [
            coeff_a[0]
            + coeff_a[1] * x
            + coeff_a[2] * x * x
            + coeff_a[3] * x * y
            + coeff_a[4] * y
            + coeff_a[5] * y * y,
            coeff_b[0]
            + coeff_b[1] * x
            + coeff_b[2] * x * x
            + coeff_b[3] * x * y
            + coeff_b[4] * y
            + coeff_b[5] * y * y,
        ]
    )


step_x = step(p)

# A and t, such that `f(x) = A*x + t` is affine transformation
A = IndexedBase("A", 4)
A = Matrix(2, 2, (A[0], A[1], A[2], A[3]))
t = IndexedBase("t", 2)
t = Matrix(2, 1, (t[0], t[1]))

# This has the effect of applying a affine transformation to the output image
step_x_affine = A**-1 * (step(A * p + t) - t)

def evidence_x_y(expr):
    return collect(expand(expr), [x, y, x * y])

step_x_affine_simpl = step_x_affine.applyfunc(evidence_x_y)

def extract_coeffs(expr):
    degress = [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]
    coeffs = []
    for i, (x_deg, y_deg) in enumerate(degress):
        coeffs.append(simplify(expr.coeff(x, x_deg).coeff(y, y_deg)))
    return Matrix([coeffs])


coeffs = step_x_affine_simpl.applyfunc(extract_coeffs)

def affine_from_pca(points):
    mean = np.median(points, axis=0)
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    pc = (points - mean) @ principal_component
    principal_length = np.max(np.abs(pc))

    perpendicular_component = np.array(
        [
            -principal_component[1],
            principal_component[0],
        ]
    )
    pc = (points - mean) @ perpendicular_component
    perpendicular_length = np.max(np.abs(pc))

    A = np.array(
        [
            principal_length * principal_component,
            perpendicular_length * perpendicular_component,
        ]
    ).T
    t = mean
    return (A, t)


class MyPrinter(RustCodePrinter):
    def _print_Mul(self, expr):

        prec = precedence(expr)

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = expr.args
        if args[0] is S.One or any(
                isinstance(a, Number) or
                a.is_Pow and all(ai.is_Integer for ai in a.args)
                for a in args[1:]):
            d, n = sift(args, lambda x:
            isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0),
                        binary=True)
            for i, di in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base

            pre = []
            # don't parenthesize first factor if negative
            if n and not n[0].is_Add and n[0].could_extract_minus_sign():
                pre = [self._print(n.pop(0))]

            nfactors = pre + [self.parenthesize(a, prec, strict=False)
                              for a in n]
            if not nfactors:
                nfactors = ['1']

            # don't parenthesize first of denominator unless singleton
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [self._print(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False)
                              for a in d]

            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        def apow(i):
            b, e = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        for item in args:
            if (item.is_commutative and
                    isinstance(item, Pow) and
                    bool(item.exp.as_coeff_Mul()[0] < 0)):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if (len(item.args[0].args) != 1 and
                            isinstance(item.base, (Mul, Pow))):
                        # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                a.append(float(item))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + "/" + b_str[0]
        else:
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)


rprinter = MyPrinter({"contract": False})

new_a = MatrixSymbol("new_a", 6, 1)
new_b = MatrixSymbol("new_b", 6, 1)
new_pos = MatrixSymbol("new_p", 2, 1)

sub_exprs, simplified_rhs = cse(coeffs)
code = ""
for var, expr in sub_exprs:
    code += "let " + rprinter.doprint(Assignment(var, expr)) + "\n"
code += "let " + rprinter.doprint(simplified_rhs[0][0].T, assign_to=new_a)
code += "let " + rprinter.doprint(simplified_rhs[0][1].T, assign_to=new_b)
code += "let " + rprinter.doprint(A.inv() * (p - t), assign_to=new_pos)

rust_template = """\
//! Module with code generated using SymPy. Check notebook for the code generator.

#[allow(non_snake_case)]
pub fn apply_affine_transform_to_attractor(a: [f64; 6], b: [f64; 6], p: [f64; 2], A: [f64; 4], t: [f64; 2]) -> ([f64;6], [f64; 6], [f64; 2]) {{
    let [x, y] = p;
{code}
    (new_a, new_b, new_p)
}}
"""

rust_code = rust_template.format(code=code)

formatted_output = subprocess.run(
    ["rustfmt", "--emit=stdout", "--edition=2021"],
    input=rust_code,
    text=True,
    stdout=subprocess.PIPE,
).stdout

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/sympy.rs"), "w") as file:
    file.write(formatted_output)

print('Done!')
