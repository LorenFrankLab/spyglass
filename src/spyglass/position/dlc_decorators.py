## dlc_decorators


def accepts(*vals, **kwargs):
    is_method = kwargs.pop("is_method", True)

    def check_accepts(f):
        if is_method:
            assert len(vals) == f.__code__.co_argcount - 1
        else:
            assert len(vals) == f.__code__.co_argcount

        def new_f(*args, **kwargs):
            pargs = args[1:] if is_method else args
            for a, t in zip(pargs, vals):  # assume first arg is self or cls
                if t is None:
                    continue
                assert a in t, "arg %r is not in %s" % (a, t)
            if is_method:
                return f(args[0], *pargs, **kwargs)
            else:
                return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_accepts
