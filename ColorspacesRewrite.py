from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import inv

'''
	Some terminological distinctions.
	
	RGB, YUV, YCbCr...  are color models. They represent colors in a specific way, 
	but do not change how the colors are perceived visually
	
	BT.709, BT.601, BT.2020, XYZ... Are color spaces. They reflect how and to what 
	degree we perceive colors visually.
	
	R4FL, UYVY, etc. are pixel formats. They are reflections of color models, and so do not 
	alter visual perception of colors. If they do, this is an error.
'''

# helper functions

# XYZ to LMS colorspace. Done for experimentation purposes. 
def  XyzToLms(inputPixel):
	matrix = [
		0.4002, 0.7076, -0.0808, 
		-0.2263, 1.1653, 0.0457, 
		0, 0, 0.9182
	]
	
	tempMatrix = np.array(matrix, dtype='<f').reshape((3,3))
	
	tempInput = np.array(matrix, dtype='<f')
	'''
		Einstein summation logic:
		
		   -> j
		| | a b c |
		v | d e f |
		i | x y z |
		
		 -> j
		| a1 b1 c1 |
		
		keep i, sum on j
	'''
	return np.einsum('ij,j->i', tempMatrix, tempInput)
	
def colorBalance(inputReferenceWhite, inputPixel):
	LMS = XyzToLms(inputPixel)
	
	referenceWhiteMatrix = np.array([1/inputReferenceWhite[0], 0, 0, 0, 1/inputReferenceWhite[1], 0, 0, 1/inputReferenceWhite[2]]).reshape((3,3))
	
	return np.einsum('ij,j->i', referenceWhiteMatrix, LMS)
	
class ColorSpace(ABC):
	def __init__(self):
		self.setPrimaries()
		self.setWhitePoint()
		self.setReferenceWhite()
		self.setBiasLevels()
		
	@abstractmethod
	def transferFunction(self, component):
		pass
	
	@abstractmethod
	def inverseTransferFunction(self, component):
		pass
		
	@abstractmethod
	def setPrimaries(self):
		pass
		
	@abstractmethod
	def setWhitePoint(self):
		pass
	
	@abstractmethod
	def setReferenceWhite(self):
		pass
		
	@abstractmethod
	def setBiasLevels(self):
		pass
		
	# get the chromaticities for colorspace 
	def generateReferenceChroma(self):
		Xr = self.xr
		Yr = self.yr
		Zr = 1 - self.xr - self.yr
		
		Xg = self.xg
		Yg = self.yg
		Zg = 1 - self.xg - self.yg
		
		Xb = self.xb
		Yb = self.yb
		Zb = 1 - self.xb - self.yb
		
		MatrixX = np.array([Xr, Xg, Xb, Yr, Yg, Yb, Zr, Zg, Zb], '>f').reshape((3,3))
		
		return MatrixX
		
	# See pg. 279 in Video Demystified book
	def getKrKgKb(self):
		MatrixX = self.generateReferenceChroma()
		
		Xw = self.xw/self.yw
		Yw = 1.0
		Zw = (1 - self.xw - self.yw)/self.yw
		
		S = np.einsum('j,ij->i', np.array([Xw, Yw, Zw], dtype='<f'), inv(MatrixX))
		
		#print('Kr = ', S[0], ', Kg = ', S[1], ', Kb = ', S[2])
		
		return S
		
	# reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	# generate the XYZ -> RGB conversion matrix based on the current colorspace's color primaries,
	# white point, and reference white
	def generateXyzConversionMatrix(self):
		# initialize chromasticities
		Xr = Xg = Xb = Yr = Yg = Yb = Zr = Zg = Zb = 0
		
		S = self.getKrKgKb()
		
		# set the chromasticities
		[[Xr, Xg, Xb], [Yr, Yg, Yb], [Zr, Zg, Zb]] = self.generateReferenceChroma().tolist()
		
		Sr = S[0]
		Sg = S[1]
		Sb = S[2]
		
		'''
				| Xr * Sr, Xg * Sg, Xb * Sb |
			M =	| Yr * Sr, Yg * Sg, Yb * Sb |
				| Zr * Sr, Zg * Sg, Zb * Sb |
		'''
		M = np.array([Xr * Sr, Xg * Sg, Xb * Sb, Yr * Sr, Yg * Sg, Yb * Sb, Zr * Sr, Zg * Sg, Zb * Sb], dtype='<f').reshape((3,3))
		
		return M
		
	def generateYuvToRgbMatrix(self):
		Kr = Kg = Kb = 0
			
		S = self.getKrKgKb()
		
		[Kr, Kg, Kb] = [S[0] * self.yr, S[1] * self.yg, S[2] * self.yb]
		
		# unquantized matrix for colorspace
		# if a colorspace does not define RGB, we can return an identity matrix
		matrixToRgb = np.array([1.0000, 0.0000, 2 * (1 - Kr), 1.0000, -2 * ((1 - Kb) * (Kb/Kg)), -2 * ((1-Kr) * (Kr/Kg)), 1.0000, 2 * (1 - Kb), 0.0000], dtype='<f').reshape((3,3))
		
		return matrixToRgb
		
	def generateRgbToYuvMatrixInverse(self):
		return inv(self.generateRgbToYuvMatrix())
		
	def generateRgbToYuvMatrix(self):
		Kr = Kg = Kb = 0

		S = self.getKrKgKb()
		
		[Kr, Kg, Kb] = [S[0] * self.yr, S[1] * self.yg, S[2] * self.yb]
		
		matrixToYuv = np.array([Kr, Kg, Kb, (1/2) * (-1 * Kr / (1-Kb)), (1/2) * (-1 * Kg/(1-Kb)), 1/2, 1/2, (1/2) * (-1 * Kg/(1-Kr)), (1/2) * (-1 * Kb/(1-Kr))], dtype='<f').reshape((3,3))
		
		return matrixToYuv
		
	# Convert native colorspace to 1931 CIE XYZ colorspace
	# Default is 8-bit linear RGB for input. You must change 
	# these for non-linear and/or YCbCr formats
	def convertToXyz(self, inputPixel, bitDepth=8, isYuv=False, isLinear=False, isQuantized=False):
		# output XYZ pixel
		XYZ = np.zeros((3))
	
		M_XYZ = self.generateXyzConversionMatrix()
		
		# Conversion steps: convert to RGB (if not RGB), the linearize (if not linear), and finally apply the XYZ conversion matrix
		
		pixel = np.array(inputPixel, dtype='<f')
		
		if (isYuv):
			max = 255 * (1 << (bitDepth - 8))
			
			RGB = self.convertToRgb(inputPixel, bitDepth, isYuv)
			
			RGB = (RGB - (self.biasR << (bitDepth - 8))) * (255 / 219) * (1 << (bitDepth - 8))
			
			for i in range(0, RGB.size):
				if RGB[i] > max:
					RGB[i] = max
			
		else:
			max = 255 * (1 << (bitDepth - 8))
		
			RGB = pixel
			
			if isQuantized:
				RGB = (RGB - (self.biasR << (bitDepth - 8))) * (255 / 219) * (1 << (bitDepth - 8))
				
				for i in range(0, RGB.size):
					if RGB[i] > max:
						RGB[i] = max
								
		print ('XYZ conversion, RGB intermediate: ', RGB)
		
		if not(isLinear):
			RGB_Linear = self.convertRgbToLinear(RGB, bitDepth)
		else:
			RGB_Linear = RGB
			
		XYZ = np.einsum('ij,j->i', M_XYZ, RGB_Linear)
		
		print("XYZ output: ", XYZ)
		
		return XYZ
	
	# override with actual bias in each colorspace
	# returns the bias of the colorspace for RGB or YUV
	def getBiasLevels(self, isYuv=False, isRgb=False):
		if(isYuv and isRgb):
			isRgb = False
			
		if(isYuv):
			return [self.biasY, self.biasCb, self.biasCr]
			
		elif(isRgb):
			return [self.biasR, self.biasG, self.biasB]
		
		else:
			# default is no bias
			return [0, 0, 0]
	
	# convert the input to RGB within the same colorspace
	'''
		Y = Kr * R + Kg * G + Kb * B
		Cb = (B - Y) / (1 - Kb) = -R * (Kr) / (1 - Kb) - G * (Kg) / (1 - Kb) + B
		Cr = (R - Y)(1- Kr) = R - G * (Kg) / (1-Kr) - B * (Kb) / (1 - Kr)
	'''
	def convertToRgb(self, inputPixel, bitDepth=8, isQuantized=True, isXyz = False):
		# output RGB
		RGB = np.zeros((3))
		
		pixel = np.array(inputPixel, dtype='<f')
		
		if not(isXyz):
			
			matrixToRgb = self.generateYuvToRgbMatrix()
			
			if(isQuantized):
				pixel = (pixel - np.array([self.biasY << (bitDepth - 8), self.biasCb << (bitDepth - 8), self.biasCr << (bitDepth - 8)], dtype='<f')) * np.array([1, 219/224, 219/224], dtype='<f')
				print(pixel)
			
			RGB = np.einsum('ij,j->i', matrixToRgb, pixel) + (self.biasR << (bitDepth - 8))
			
		else:
			# inverse XYZ -> RGB matrix
			M_Inverse = inv(self.generateXyzConversionMatrix())
			
			# linear RGB from matrix conversion
			RGB_Linear = np.einsum('ij,j->i', M_Inverse, pixel)
			
			for i in range(0, RGB_Linear.size):
				if RGB_Linear[i] > 1:
					RGB_Linear[i] = 1
			
			# gamma correct and quantize
			RGB = self.convertRgbToGammaCorrected(RGB_Linear, bitDepth) + (self.biasR << (bitDepth - 8))
			
		print("RGB output: ", RGB)
		
		return RGB
	
	# convert the input to RGB within the same colorspace
	'''
		R = Y + 0 * U + V(1-Kr)
		G = Y - U(1 - Kb)(Kb / Kg) -V(1 - Kr)(Kr / Kg)
		B = Y + U(1 - Kb) + 0 * V 
	'''
	def convertToYuv(self, inputPixel, bitDepth=8, isXyz=False, isLinear=False, isQuantized=True):
		# output YUV
		YUV = np.zeros((3))
	
		# convert input list to numpy array
		pixel = np.array(inputPixel, dtype='<f')
	
		matrixToYuv = self.generateRgbToYuvMatrix()
	
		print(matrixToYuv)
		
		if isXyz:
			# inverse XYZ -> RGB matrix
			M_inverse = inv(self.generateXyzConversionMatrix())
			
			# linear RGB from matrix conversion
			RGB_Linear = np.einsum('ij,j->i', M_inverse, pixel)
			
			for i in range(0, RGB_Linear.size):
				if RGB_Linear[i] > 1:
					RGB_Linear[i] = 1
			
			# gamma  correct and quantize
			RGB_gamma = self.convertRgbToGammaCorrected(RGB_Linear, bitDepth) + (self.biasR << (bitDepth - 8))
			
		else:
			if isLinear:
				# gamma correct for colorspace
				RGB_gamma = self.convertRgbToGammaCorrected(pixel, bitDepth)
				
			else:
				# output is already gamma corrected
				RGB_gamma = pixel
			
			if not(isQuantized):
				# need to quantize the output for most colorspaces
				RGB_gamma = (RGB_gamma * 219 + self.biasR) * (2**(bitDepth - 8))
		
		YUV = np.einsum('ij,j->i', matrixToYuv, RGB_gamma)
		
		YUV = (YUV * np.array([1, 224/219, 224/219], dtype='<f')) + np.array([0, self.biasCb << (bitDepth -8), self.biasCr << (bitDepth - 8)], dtype='<f')

		print("YUV output:", YUV)

		return YUV
		
	def convertRgbToLinear(self, inputPixel, bitDepth):
		pixel = np.array(inputPixel, dtype='<f') / ((1 << bitDepth) - 1)
		
		print ("Normalized Pixel: ", pixel)
		
		pixel = np.vectorize(self.inverseTransferFunction)(pixel)
		
		print ("Linear Pixel: ", pixel)
		
		return  pixel
		
	def convertRgbToGammaCorrected(self, inputPixel, bitDepth):
		pixel = np.vectorize(self.transferFunction)(np.array(inputPixel, dtype='<f'))
	
		print("Gamma Pixel: ", pixel)
		
		pixel = pixel * ((1 << bitDepth) - 1)
		
		print ("Scaled Pixel: ", pixel)
		
		return  pixel

# BT.709 colorspace. Inherits from Colorspace		
class BT709(ColorSpace):
	def __init__(self):
		super().__init__()
	
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component < 0.018):
			correctedComponent = component * 4.500
		
		elif (component >= 0.018):
			correctedComponent = 1.099 * pow(component, 0.45) - 0.099
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component < 0.081):
			linearComponent = component / 4.500
		
		elif (component >= 0.081):
			linearComponent = pow((component + 0.099)/1.099, 1/0.45)
		
		return linearComponent
		
	def setPrimaries(self):
		self.xr = 0.64
		self.yr = 0.33
		self.xg = 0.30
		self.yg = 0.60
		self.xb = 0.15
		self.yb = 0.06
		
	def setWhitePoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
	
	def setReferenceWhite(self):
		self.Xw = 0.2126
		self.Yw = 0.7152
		self.Zw = 0.0722
		
	def setBiasLevels(self):
		self.biasY = 16
		self.biasCb = 128
		self.biasCr = 128
		
		self.biasR = 16
		self.biasG = 16
		self.biasB = 16

# BT.601 Colorspace, 625 Line System. 
class BT601625(ColorSpace):
	def __init__(self):
		super().__init__()
		
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component < 0.018):
			correctedComponent = component * 4.500
		
		elif (component >= 0.018):
			correctedComponent = 1.099 * pow(component, 0.45) - 0.099
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component < 0.081):
			linearComponent = component / 4.500
		
		elif (component >= 0.081):
			linearComponent = pow((component + 0.099)/1.099, 1/0.45)
		
		return linearComponent	
	
	def setPrimaries(self):
		self.xr = 0.64
		self.yr = 0.33
		self.xg = 0.29
		self.yg = 0.60
		self.xb = 0.15
		self.yb = 0.06
	
	def setWhitePoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
		
	def setReferenceWhite(self):
		self.Xw = 0.2126
		self.Yw = 0.7152
		self.Zw = 0.0722
		
	def setBiasLevels(self):
		self.biasY = 16
		self.biasCb = 128
		self.biasCr = 128
		
		self.biasR = 16
		self.biasG = 16
		self.biasB = 16

# BT.601 Colorspace, 525 Line System. Inherits from BT.709 because they share the same transfer function		
class BT601525(ColorSpace):
	def __init__(self):
		super().__init__()
	
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component < 0.018):
			correctedComponent = component * 4.500
		
		elif (component >= 0.018):
			correctedComponent = 1.099 * pow(component, 0.45) - 0.099
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component < 0.081):
			linearComponent = component / 4.500
		
		elif (component >= 0.081):
			linearComponent = pow((component + 0.099)/1.099, 1/0.45)
		
		return linearComponent	
	
	def setPrimaries(self):
		self.xr = 0.63
		self.yr = 0.34
		self.xg = 0.31
		self.yg = 0.595
		self.xb = 0.155
		self.yb = 0.07
	
	def setWhitePoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
		
	def setReferenceWhite(self):
		self.Xw = 0.2126
		self.Yw = 0.7152
		self.Zw = 0.0722
		
	def setBiasLevels(self):
		self.biasY = 16
		self.biasCb = 128
		self.biasCr = 128
		
		self.biasR = 16
		self.biasG = 16
		self.biasB = 16

class BT601(ColorSpace):
	def __init__(self):
		super().__init__()
	
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component < 0.018):
			correctedComponent = component * 4.500
		
		elif (component >= 0.018):
			correctedComponent = 1.099 * pow(component, 0.45) - 0.099
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component < 0.081):
			linearComponent = component / 4.500
		
		elif (component >= 0.081):
			linearComponent = pow((component + 0.099)/1.099, 1/0.45)
		
		return linearComponent	
	
	def setPrimaries(self):
		self.xr = 0.67
		self.yr = 0.33
		self.xg = 0.21
		self.yg = 0.71
		self.xb = 0.14
		self.yb = 0.08
	
	def setWhitePoint(self):
		self.xw = 0.3101
		self.yw = 0.3162
		
	def setReferenceWhite(self):
		self.Xw = 0.2126
		self.Yw = 0.7152
		self.Zw = 0.0722
		
	def setBiasLevels(self):
		self.biasY = 16
		self.biasCb = 128
		self.biasCr = 128
		
		self.biasR = 16
		self.biasG = 16
		self.biasB = 16
		
# sRGB Colorspace. Shares the BT.709 primaries, but not the transfer characteristic.
class sRGB(ColorSpace):
	def __init__(self):
		super().__init__()
	
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component <= 0.0031308):
			correctedComponent = component * 12.92
			
		elif (component > 0.0031308):
			correctedComponent = 1.055 * pow(component, 1/2.4) - 0.055
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component <= 0.04045):
			linearComponent = component / 12.92
		
		elif (component > 0.04045):
			linearComponent = pow((component + 0.055)/1.055, 2.4)
		
		return linearComponent
		
	def setPrimaries(self):
		self.xr = 0.64
		self.yr = 0.33
		self.xg = 0.30
		self.yg = 0.60
		self.xb = 0.15
		self.yb = 0.06
		
	def setWhitePoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
	
	def setReferenceWhite(self):
		self.Xw = 0.2126
		self.Yw = 0.7152
		self.Zw = 0.0722
		
	def setBiasLevels(self):
		self.biasY = 0
		self.biasCb = 0
		self.biasCr = 0
		
		self.biasR = 0
		self.biasG = 0
		self.biasB = 0
		
class AdobeRgb(ColorSpace):
	
	def transferFunction(self, component):
		correctedComponent = 0.0
		
		if (component >= 0) and (component <= 1):
			correctedComponent = pow(component, 1/2.19921875)
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		linearComponent = 0.0
		
		if (component >= 0) and (component <= 1):
			linearComponent = pow(component, 2.19921875)
		
		return linearComponent
	
	def setPrimaries(self):
		self.xr = 0.6400
		self.yr = 0.3300
		self.xg = 0.2100
		self.yg = 0.7100
		self.xb = 0.1500
		self.yb = 0.0600
	
	def setWhitepoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
		
class BT2020(ColorSpace):
	def transferFunction(self, component):
		alpha = 1.09929682680944
		
		beta = 0.018053968510807
		
		correctedComponent = 0.0
		
		if (component >= 0) and (component < beta):
			correctedComponent = 4.5 * component
			
		elif (component >= beta):
			correctedComponent = (alpha * pow(component, 0.45)) - (alpha - 1)
		
		return correctedComponent
	
	def inverseTransferFunction(self, component):
		alpha = 1.09929682680944
		
		beta = 0.018053968510807
		
		linearComponent = 0.0
		
		if (component >= 0) and (component < beta):
			linearComponent = component / 4.5
			
		elif (component >= beta):
			linearComponent = pow((component + alpha - 1) / alpha, 1/0.45)
			
		return linearComponent
	
	def setPrimaries(self):
		self.xr = 0.708
		self.yr = 0.292
		self.xg = 0.170
		self.yg = 0.797
		self.xb = 0.131
		self.yb = 0.046
	
	def setWhitePoint(self):
		self.xw = 0.3127
		self.yw = 0.3290
		
	def setReferenceWhite(self):
		self.Xw = 0.89965814 
		self.Yw = 0.8506877
		self.Zw = 1.2891678
		
	def setBiasLevels(self):
		self.biasY = 16
		self.biasCb = 128
		self.biasCr = 128
		
		self.biasR = 16
		self.biasG = 16
		self.biasB = 16
	