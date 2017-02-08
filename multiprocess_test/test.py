from multiprocess import Pool
p = Pool(4)
print (p.map(lambda x: (lambda y:y**2)(x) + x, xrange(10)))