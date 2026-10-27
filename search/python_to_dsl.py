import ast

from search.expressions import (
    InputExpr,
    ConstantExpr,
    PrimitiveExpr,
    PrimitiveRefExpr,
)


def parse_expression(node: ast.expr, registry: dict):
    """Convertit une expression AST Python en expression DSL."""
    if isinstance(node, ast.Name):
        name = node.id

        if name == "I":
            return InputExpr()

        if name in registry["constants"]:
            constant = registry["constants"][name]
            return ConstantExpr(
                name,
                constant["value"],
                constant["type"],
            )

        if name in registry["primitives"]:
            return PrimitiveRefExpr(name)

        raise ValueError(f"Nom inconnu : {name}")

    if isinstance(node, ast.Call):
        return parse_primitive_call(node, registry)

    raise ValueError(
        f"Expression Python non prise en charge : {ast.dump(node)}"
    )


def parse_primitive_call(node: ast.Call, registry: dict):
    """Convertit un appel de primitive Python en expression DSL."""
    if not isinstance(node.func, ast.Name):
        raise ValueError("Seuls les appels directs à une fonction sont acceptés.")

    name = node.func.id

    if name not in registry["primitives"]:
        raise ValueError(f"Primitive inconnue : {name}")

    arguments = tuple(
        parse_expression(argument, registry)
        for argument in node.args
    )

    return PrimitiveExpr(name, arguments, None)


def parse_expression_with_variables(
    node: ast.expr,
    registry: dict,
    variables: dict,
):
    """Convertit une expression AST en résolvant les variables locales."""
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]

    if isinstance(node, ast.Call):
        # Appel d'une fonction stockée dans une variable locale : x2(I)
        if isinstance(node.func, ast.Name) and node.func.id in variables:
            function = variables[node.func.id]

            arguments = tuple(
                parse_expression_with_variables(arg, registry, variables)
                for arg in node.args
            )

            from search.expressions import ApplyExpr

            return ApplyExpr(function=function, arguments=arguments)

        if not isinstance(node.func, ast.Name):
            raise ValueError("Seuls les appels directs à une fonction sont acceptés.")

        name = node.func.id

        if name not in registry["primitives"]:
            raise ValueError(f"Primitive inconnue : {name}")

        arguments = tuple(
            parse_expression_with_variables(argument, registry, variables)
            for argument in node.args
        )

        return PrimitiveExpr(name, arguments, None)

    return parse_expression(node, registry)


def parse_solver_body(function_node: ast.FunctionDef, registry: dict):
    """Convertit les affectations et le return d'un solveur en expression DSL."""
    variables = {}

    for statement in function_node.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                raise ValueError("Affectation non prise en charge.")

            name = statement.targets[0].id
            variables[name] = parse_expression_with_variables(
                statement.value, registry, variables
            )

        elif isinstance(statement, ast.Return):
            return parse_expression_with_variables(
                statement.value, registry, variables
            )

        else:
            raise ValueError(
                f"Instruction Python non prise en charge : "
                f"{ast.dump(statement)}"
            )

    raise ValueError("Le solveur ne contient aucune instruction return.")


def parse_solver(source: str, registry: dict):
    """Parse le code source d'un solveur et retourne son expression DSL."""
    module = ast.parse(source)

    solver_functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("solve_")
    ]

    if len(solver_functions) != 1:
        raise ValueError(
            f"Un seul solveur attendu, {len(solver_functions)} trouvé(s)."
        )

    return parse_solver_body(solver_functions[0], registry)
