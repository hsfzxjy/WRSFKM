from init import init_uv


def run(X, labels, p, logger):

    C = p.C

    assert isinstance(p.iter_method, str)

    if p.init == 'preset':
        U, V = p.initial
    else:
        U, V = init_uv(X, C, method=p.init)

    iter_method = p.iter_method.lower()

    if iter_method in ['sv', 'mv']:
        from iterations.normal import iteration
    elif iter_method == 'aa':
        from iterations.anderson import iteration
    elif iter_method == 'orig':
        from iterations.orig import iteration
    elif iter_method == 'kmeans':
        from iterations.kmeans import iteration
    elif iter_method == 'fkm':
        from iterations.fkm import iteration
    else:
        raise RuntimeError('Unknown method.')

    return iteration(X, U, V, labels, p, logger=logger)
