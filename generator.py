def generate_int(N):
    for i in xrange(N):
        yield i

def factorial(x):
    if x > 0:
        return x*factorial(x-1)
    else:
        return 1

print factorial(3)
print factorial(10)
