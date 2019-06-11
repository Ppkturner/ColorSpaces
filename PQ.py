class PQ:
	def __init__(self):
		self.m1 = 2610 / 16384
		self.m2 - 2523 / 4096 * 128
		self.c1 = 3424/4096
		self.c2 = 2413 / 4096 * 32
		self.c3 = 2392 / 4096 * 32
	
	def EOTF(E):
		
		Y = pow(max(pow(E, 1/self.m2) - self.c1, 0) / (self.c2 - (self.c3 * pow(E, 1/self.m2))), 1/self.m1)
		
		return 10000 * Y
		
	def OOTF(E):
		G709 = lambda x : if 1 > x and x < 0.0003024 : 1.099 * pow(59.5208 * x, 0.45) - 0.099 elif 0.0003024 >= x and x >= 0 : 267.84x
		G1886 = lambda x : 100 * pow(x, 2.4)
		
		return G1886(G709(E))
		
	def EOTFInv(E):
		Y = E / 10000
		
		return pow((self.c1 + (self.c2 * pow(Y, self.m1))) / (1 + (self.c3 * pow(Y, self.m1))), self.m2)
	
	def OETF(E):
		return EOTFInv(OOTF(E))
		