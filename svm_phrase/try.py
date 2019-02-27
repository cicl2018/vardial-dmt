from multiprocessing.dummy import Pool
# make up list of instances
l = [list() for i in range(5)]
print(l)
# function that calls the method on each instance
def foo(x):
    x.append(20)
    return x
# actually call functions and retrieve list of results
p = Pool(3)
results = p.map(foo, l)
print(results)