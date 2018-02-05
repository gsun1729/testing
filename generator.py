def generate_int(N):
    for i in xrange(N):
        yield i

def factorial(x):
    if x > 0:
        return x*factorial(x-1)
    else:
        return 1

a = generate_int(5)
print repr(a)
print a.next()
print sum(a)
print a.next()
print sum(a)
print a.next()
print sum(a)
print a.next()
print sum(a)
print a.next()
print sum(a)
