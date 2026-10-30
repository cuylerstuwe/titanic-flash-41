def plan(targets, dependencies, completed=()):
    """Return the work needed to build the requested targets."""
    answer = []
    seen = set(completed)

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for dependency in dependencies.get(node, ()):
            visit(dependency)
        answer.append(node)

    for target in targets:
        visit(target)
    return answer
