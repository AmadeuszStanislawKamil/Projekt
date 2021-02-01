importanceRadiation = 1
importanceAmbient = 0
importanceModule = 0


WeatherRadiationFile = open("weatherRadiation.csv", "r")

weatherRadiationString = WeatherRadiationFile.readlines()

weatherX0 = []

for x in weatherRadiationString:
    weatherX0.append(float(x.rstrip('\n')))

print(weatherX0)

WeatherAmbientFile = open("weatherAmbient.csv", "r")

weatherAmbientString = WeatherAmbientFile.readlines()

weatherX1 = []

for x in weatherAmbientString:
    weatherX1.append(float(x.rstrip('\n')))

print(weatherX1)

WeatherModuleFile = open("weatherModule.csv", "r")

weatherModuleString = WeatherModuleFile.readlines()

weatherX2 = []

for x in weatherModuleString:
    weatherX2.append(float(x.rstrip('\n')))

print(weatherX2)


DcFile = open("dc.csv","r")

dcString = DcFile.readlines()

dcY = []

for x in dcString:
    dcY.append(float(x.rstrip('\n')))

print(dcY)


weatherX0Square = []
weatherX1Square = []
weatherX2Square = []

for x in weatherX0:
    weatherX0Square.append(x*x)

print(weatherX0Square)

for x in weatherX1:
    weatherX1Square.append(x*x)

print(weatherX1Square)

for x in weatherX2:
    weatherX2Square.append(x*x)

print(weatherX2Square)


XY0 = []
XY1 = []
XY2 = []

for x in range (len(weatherX0)):
    XY0.append(dcY[x]*weatherX0[x])

print(XY0)

for x in range (len(weatherX1)):
    XY1.append(dcY[x]*weatherX1[x])

print(XY1)

for x in range (len(weatherX2)):
    XY2.append(dcY[x]*weatherX2[x])

print(XY2)


SUMX0 = 0
SUMX1 = 0
SUMX2 = 0

SUMX0Square = 0
SUMX1Square = 0
SUMX2Square = 0

SUMXY0 = 0
SUMXY1 = 0
SUMXY2 = 0

SUMY = 0


for x in weatherX0:
    SUMX0+=x

print(SUMX0)

for x in weatherX1:
    SUMX1+=x

print(SUMX1)

for x in weatherX2:
    SUMX2+=x

print(SUMX2)

for x in weatherX0Square:
    SUMX0Square+=x

print(SUMX0Square)

for x in weatherX1Square:
    SUMX1Square+=x

print(SUMX1Square)

for x in weatherX1Square:
    SUMX1Square+=x

print(SUMX1Square)

for x in XY0:
    SUMXY0+=x

print(XY0)

for x in XY1:
    SUMXY1+=x

print(XY1)

for x in XY1:
    SUMXY1+=x

print(XY1)

for x in dcY:
    SUMY+=x

print(SUMY)

m0 = 0

m1 = 0

m2 = 0

N=len(weatherX0)

print("N: "+str(N))

m0 = (N * SUMXY0 - SUMX0 * SUMY )/ (N * SUMX0Square - (SUMX0*SUMX0))
print("m0: "+str(m0))

m1 = (N * SUMXY1 - SUMX1 * SUMY )/ (N * SUMX1Square - (SUMX1*SUMX1))
print("m1: "+str(m1))

m2 = (N * SUMXY2 - SUMX2 * SUMY )/ (N * SUMX2Square - (SUMX2*SUMX2))
print("m2: "+str(m2))


b0 = 0
b1 = 0
b2 = 0

b0 = (SUMY - m0 * SUMX0) / N
print("b0: "+str(b0))

b1 = (SUMY - m1 * SUMX1) / N
print("b1: "+str(b1))

b2 = (SUMY - m2 * SUMX2) / N
print("b2: "+str(b2))

b = b0+b1+b2
print("b: "+str(b))

print("y="+str(b)+"+"+str(m0)+"*x+"+str(m1)+"*x+"+str(m2)+"*x")


WeatherRadiationCalculations = open("weatherRadiationCalculations.csv", "r")

weatherRadiationStringCalculations = WeatherRadiationCalculations.readlines()

weatherX0Calculations = []

for x in weatherRadiationStringCalculations:
    weatherX0Calculations.append(float(x.rstrip('\n')))

print(weatherX0Calculations)

WeatherAmbientFileCalculations = open("weatherAmbientCalculations.csv", "r")

weatherAmbientStringCalculations = WeatherAmbientFileCalculations.readlines()

weatherX1Calculations = []

for x in weatherAmbientStringCalculations:
    weatherX1Calculations.append(float(x.rstrip('\n')))

print(weatherX1Calculations)

WeatherModuleFileCalculations = open("weatherModuleCalculations.csv", "r")

weatherModuleStringCalculations = WeatherModuleFileCalculations.readlines()

weatherX2Calculations = []

for x in weatherModuleStringCalculations:
    weatherX2Calculations.append(float(x.rstrip('\n')))

print(weatherX2Calculations)


dcFile2 = open("dcCalculations.csv", "w")
print("Calculations")
dcYCalculated = []
for x in range (len(weatherX2Calculations)):
    dcOutput=b+importanceRadiation*(m0*float(weatherRadiationStringCalculations[x]))+importanceAmbient*(m1*float(weatherAmbientStringCalculations[x]))+importanceModule*(m2*float(weatherModuleStringCalculations[x]))
    if dcOutput<0:
        dcOutput=0
    dcYCalculated.append(dcOutput)
    dcFile2.write(str(dcOutput) + '\n')


print(dcYCalculated)




"""
X2 = []
for x in weatherX:
    X2.append(float(x)*float(x))

print(X2)


DcFile = open("dc.csv","r")

dcString = DcFile.readlines()

dcY = []

for x in dcString:
    dcY.append(float(x.rstrip('')))

print(dcY)

XY = []
for x in range (len(weatherX)):
    XY.append(float(weatherX[x])*float(dcY[x]))

print(XY)

SumX = 0
for x in weatherX:
    SumX+=x

print(SumX)

SumY = 0
for x in dcY:
    SumY+=x

print(SumY)

SumX2 = 0
for x in X2:
    SumX2 += x

print(SumX2)

SumXY = 0
for x in XY:
    SumXY += x

print(SumXY)

#N - number of points

N=len(weatherX)


#Slope M

M = (N * SumXY - SumX * SumY )/ (N * SumX2 - (SumX*SumX))

print(M)

#Intercept B

B = (SumY - M*SumX) / N

print(B)


print("y ="+str(M)+"*x"+str(B))

print("results")
print("orygynal")
for x in dcY:
    print(x,end=', ')

dcFile2 = open("dcCalculations.csv", "w")
weatherFile2 = open("weatherCalculations.csv","r")

weatherCalculations = []
weatherCalculations = weatherFile2.readlines()


weatherCalculations2 = []
for x in weatherCalculations:
    weatherCalculations2.append(float(x.rstrip('')))

print()
print("Calculations")
for x in weatherCalculations2:
    calculations = M * x + B
    if calculations<0:
        calculations=0
    print(str(calculations),end=', ')
    dcFile2.write(str(calculations)+'')

WeatherFile.close()
DcFile.close()
dcFile2.close()
weatherFile2.close()

"""