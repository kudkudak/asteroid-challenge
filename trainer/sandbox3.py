from utils import cached_HDD


@cached_HDD()
def f():
    return [1,2,3]


print f()

print f()