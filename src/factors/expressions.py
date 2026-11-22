"""Validate a deliberately small Qlib-style expression DSL; never execute model code."""
import ast
import math
import re

FIELDS = {'open', 'high', 'low', 'close', 'volume', 'vwap', 'amount'}
ROLLING = {'Mean', 'Std', 'Slope', 'Rsquare', 'Resi', 'Max', 'Min', 'Rank', 'IdxMax', 'IdxMin', 'Sum'}


def inspect_expression(expression):
    if not isinstance(expression, str) or not 0 < len(expression) <= 4096:
        raise ValueError('Expression must contain 1..4096 characters')
    fields = set(re.findall(r'\$([a-z_]+)', expression))
    if not fields or not fields <= FIELDS:
        raise ValueError('Unsupported or missing input fields')
    try:
        tree = ast.parse(re.sub(r'\$([a-z_]+)', r'field_\1', expression), mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid expression syntax') from exc
    if len(list(ast.walk(tree))) > 400:
        raise ValueError('Expression too complex')

    def window(node, minimum=1):
        if not isinstance(node, ast.Constant) or type(node.value) is not int or not minimum <= node.value <= 252:
            raise ValueError('Window must be a literal integer in range; future references forbidden')
        return node.value

    def visit(node):
        if isinstance(node, ast.Name) and node.id.startswith('field_') and node.id[6:] in fields:
            return 0
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and math.isfinite(node.value):
            return 0
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return max(visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Gt, ast.Lt, ast.GtE, ast.LtE)):
            return max(visit(node.left), visit(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name, args = node.func.id, node.args
            if name in {'Abs', 'Log'} and len(args) == 1:
                return visit(args[0])
            if name in {'Greater', 'Less'} and len(args) == 2:
                return max(map(visit, args))
            if name == 'Ref' and len(args) == 2:
                return visit(args[0]) + window(args[1], 0)
            if name in ROLLING and len(args) == 2:
                return visit(args[0]) + window(args[1]) - 1
            if name == 'Corr' and len(args) == 3:
                return max(visit(args[0]), visit(args[1])) + window(args[2]) - 1
            if name == 'Quantile' and len(args) == 3:
                q = args[2]
                if not isinstance(q, ast.Constant) or type(q.value) not in (int, float) or not 0 <= q.value <= 1:
                    raise ValueError('Invalid quantile')
                return visit(args[0]) + window(args[1]) - 1
        raise ValueError('Unsupported expression node or operator')

    history = visit(tree.body)
    # Catch unprefixed names even when they coincide with internally translated fields.
    if re.search(r'(?<!\$)\bfield_', expression):
        raise ValueError('Fields must use $ prefix')
    return sorted(fields), history + 1, ast.dump(tree, include_attributes=False)
