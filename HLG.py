import math
import matplotlib.pyplot as plt

class HLG:
	def __init__(self, Lw, Lb):
		self.Lw = Lw
		
		self.Lb = Lb
		
		self.a = 0.17883277
		self.b = 0.28466892
		self.c = 0.55991073
		
		self.gamma = self.GetGamma()

	def GetGamma(self):
		gamma = 1.2
		kappa = 1.111
		
		if self.Lw > 1000:
			gamma = 1.2 + 0.42 * (math.log(self.Lw / 1000) / math.log(10))
		
		elif gamma < 400 or gamma > 2000:
			gamma = 1.2 * kappa ** (math.log(self.Lw / 1000) / math.log(2))

		return gamma
		

	def OETF(self, E):
		output = []
		
		for e in E:
			if e > 1.0:
				e = 1.0
				
			elif e < 0.0:
				e = 0.0
		
			if e >= 0 and e <= (1 / 12):
				output.append(math.sqrt(3 * e))
			
			elif e <= 1:
				output.append(self.a * math.log((12 * e) - self.b) + self.c)
				
		return output
		
	def OETFInv(self, E):	
		output = []
		
		for e in E:
			if e > 1.0:
				e = 1.0
				
			elif e < 0.0:
				e = 0.0
		
			if e >= 0 and e <= 0.5:
				output.append((e ** 2) / 3)
			
			elif e <= 1:
				output.append((math.exp((e - self.c) / self.a) + self.b) / 12)
				
		print("OETF Inverse: ", output)
		
		return output
			
	def OOTF(self, E):
		alpha = 1.0
		
		[R, G, B] = E
		
		# This is actually luminance
		Y = 0.2627 * R + 0.6780 * G + 0.0593 * B
			
		Rd = alpha * (Y ** (self.gamma - 1)) * R
		Gd = alpha * (Y ** (self.gamma - 1)) * G
		Bd = alpha * (Y ** (self.gamma - 1)) * B
		
		print("OOTF: ", Rd, Gd, Bd)
		
		return [Rd, Gd, Bd]
		
	def OOTFInv(self, E):
		alpha = 1.0
		
		[Rd, Gd, Bd] = E
		
		Y = 0.2627 * Rd + 0.6780 * Gd + 0.0593 * Bd
		
		R = (Y / alpha) ** ((1 - self.gamma) / self.gamma) * (Rd / alpha)
		G = (Y / alpha) ** ((1 - self.gamma) / self.gamma) * (Gd / alpha)
		B = (Y / alpha) ** ((1 - self.gamma) / self.gamma) * (Bd / alpha)
		
		print ("OOTF Inverse: ", R, G, B)
		
		return [R, G, B]

	# From Video Signal to Display signal
	def EOTF(self, E):
		beta = math.sqrt(3 * ((self.Lb / self.Lw) ** (1 / self.gamma)))
		
		output = []
		
		for e in E:
			output.append(max(0, (1 - beta) * e + beta))
			
		output = self.OOTF(self.OETFInv(output))
		
		print ("EOTF: ", output)
		
		return output

	# From Display signal to video signal
	def EOTFInv(self, E):

		output = self.OETF(self.OOTFInv(E))

		print ("EOTF Inverse", output)
		
		return output
	
def plotTransferFunction():
	
	val = 0.0
	
	valuesEOTF = []
	
	valuesOETF = []
	
	valuesEOTFInv = []
	
	valuesEotfEotfInv = []
	
	valuesLinear = []
	
	y = []
	
	points = 10000
	
	for i in range(0, points):
		
		valuesEOTF.append(EOTF([val, val, val])[0])
		
		valuesOETF.append(OETF([val, val, val])[0])
		
		valuesEOTFInv.append(EOTFInv([val, val, val])[0])
		
		valuesEotfEotfInv.append(EOTFInv(EOTF([val, val, val])[0]))
		
		valuesLinear.append(val)
		
		val += 1/(points)
		
		y.append(val);
		
	plt.plot(y, valuesEOTF, label="EOTF")
	
	plt.plot(y, valuesOETF, label="OETF")
	
	plt.plot(y, valuesEOTFInv, label="EOTF Inverse")
	
	plt.plot(y, valuesEotfEotfInv, label="EOTF -> EOTF Inverse")
	
	plt.plot(y, valuesLinear, label="Linear")
	
	plt.legend()
	
	plt.show()