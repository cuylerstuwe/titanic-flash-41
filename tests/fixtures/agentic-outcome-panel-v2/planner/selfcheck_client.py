"""Known-good control for the explicit fixture contract; never shown to the model."""
import heapq


def plan(targets, dependencies, completed=()):
    done = set(completed)
    needed = set()
    todo = list(targets)
    while todo:
        node = todo.pop()
        if node in needed or node in done:
            continue
        needed.add(node)
        todo.extend(dependencies.get(node, ()))
    indegree = {node: 0 for node in needed}
    dependents = {node: [] for node in needed}
    for node in needed:
        for dep in set(dependencies.get(node, ())):
            if dep in needed:
                indegree[node] += 1
                dependents[dep].append(node)
    ready = [node for node in needed if indegree[node] == 0]
    heapq.heapify(ready)
    answer = []
    while ready:
        node = heapq.heappop(ready)
        answer.append(node)
        for dependent in dependents[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(answer) != len(needed):
        raise ValueError('cycle in unfinished reachable graph')
    return answer
