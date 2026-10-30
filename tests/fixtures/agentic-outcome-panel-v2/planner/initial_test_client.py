from client import plan


def test_dependencies_first():
    assert plan(['app'], {'app': ['lib']}) == ['lib', 'app']


def test_smallest_ready_node():
    assert plan(['z', 'a'], {}) == ['a', 'z']
