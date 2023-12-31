import streamlit as st
st.set_page_config(page_title = "Gas Dynamics Calculator", layout = "wide")
st.title("Gas Dynamics Calculator")
st.write("""
 - If you get garbage values, make sure that your values are not of bounds
 - This calculator is made for the assumption of perfect gas, whose specific heats don't change with temperature. This assumption breaks for values of M > 8""")

#ISENTROPIC FLOW

st.subheader("Isentropic flow")
g1 = st.number_input('The value of gamma is ', min_value = 1.001, key = "is")
from scipy.optimize import fsolve
def isd_ratio(M,g):
    return (1 + ((g-1)/2)*M*M)**(1/(1-g))
def isT_ratio(M,g):
    return (1 + ((g-1)/2)*M*M)**(-1)
def isP_ratio(M,g):
    return (1 + ((g-1)/2)*M*M)**(g/(1-g))
def isa_ratio(M,g):
    return (isT_ratio(M,g))**(0.5)
def isA_ratio(M,g):
    return abs((1/M)*(((2/(g+1))*(1 + ((g-1)/2)*M*M))**((g+1)/(2*(g-1)))))
def isMach(M,g):
    return ((g+1)*M*M/((g-1)*M*M + 2))**(0.5)
def isshow(M,g):
    st.write("M = " + str(round(abs(M),4)))
    st.write("p/p0 = " + str(round(isP_ratio(M,g1),4))) 
    st.write("T/T0 = " + str(round(isT_ratio(M,g1),4)))
    st.write("rho/rho0 = " + str(round(isd_ratio(M,g1),4)))
    st.write("A/A* = " + str(round(isA_ratio(M,g1),4))) 
    st.write("a/a0 = " + str(round(isa_ratio(M,g1),4)))
    st.write("M* = " + str(round(isMach(M,g1),4)))
s1 = st.selectbox('Select the variable',('M', 'p/p0', 'T/T0','rho/rho0','A/A*','a/a0','M*'), key = "is1")
z = ((g1+1)/(g1-1))
if s1 == "M":
    x = st.number_input('Enter', min_value = 0.001)
    isshow(x,g1)
elif s1 == "T/T0":
    x = st.number_input('Enter', max_value = 1.0)
    x = float(x)
    def f(z):
        M = z
        f0 = isT_ratio(M,g1) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    isshow(M,g1)
elif s1 == "p/p0":
    x = st.number_input('Enter', max_value = 1.0)
    def f(z):
        M = z
        f0 = isP_ratio(M,g1) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    isshow(M,g1)
elif s1 == "a/a0":
    x = st.number_input('Enter', max_value = 1.0)
    def f(z):
        M = z
        f0 = isa_ratio(M,g1) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    isshow(M,g1)
elif s1 == "A/A*":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = isA_ratio(M,g1) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    isshow(M,g1)
    def f(z):
        M = z
        f0 = isA_ratio(M,g1) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    isshow(M,g1)
elif s1 == "rho/rho0":
    x = st.number_input('Enter', max_value = 1.0)
    def f(z):
        M = z
        f0 = isd_ratio(M,g1) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    isshow(M,g1)
elif s1 == "M*":
    x = st.number_input('Enter', max_value = z)
    def f(z):
        M = z
        f0 = isMach(M,g1) - x
        return f0
    zGuess = 0.1
    z = fsolve(f,zGuess)
    M = float(z)
    isshow(M,g1)

#NORMAL SHOCK

st.subheader("Normal Shock")
g5 = st.number_input('The value of gamma is ', min_value = 1.001, key = "no")
def noM2(M1,g):
    return ((1 + 0.5*(g-1)*M1*M1)/(g*M1*M1 - 0.5*(g-1)))**(0.5)
def nod_ratio(M1,g):
    return ((g+1)*M1*M1)/((g-1)*M1*M1 + 2) 
def noP_ratio(M1,g):
    return 1 + (2*g)*(M1*M1 - 1)/(g+1)
def noT_ratio(M1,g):
    return (1 + 2*(g-1)*(g*M1*M1 + 1)*(M1*M1 - 1)/(M1*M1*(g+1)*(g+1)))
def noPs_ratio(M1,g):
    return ((1 + (2*g)*(M1*M1 - 1)/(g+1))**(1/(1-g)))*((((g+1)*M1*M1)/((g-1)*M1*M1 + 2))**(g/(g-1)))
def noa_ratio(M1,g):
    return (abs(1 + 2*(g-1)*(g*M1*M1 + 1)*(M1*M1 - 1)/(M1*M1*(g+1)*(g+1))))**(0.5)
def noshow(M1,g):
    st.write("M1 = " + str(abs(M1)))
    st.write("M2 = " + str(noM2(M1,g))) 
    st.write("P2/P1 = " + str(noP_ratio(M1,g)))
    st.write("rho2/rho1 = " + str(nod_ratio(M1,g)))
    st.write("T2/T1 = " + str(noT_ratio(M1,g))) 
    st.write("a2/a1 = " + str(noa_ratio(M1,g)))
    st.write("p02/p01 = " + str(noPs_ratio(M1,g)))
s5 = st.selectbox('Select the variable',('M1', 'M2', 'P2/P1','rho2/rho1','T2/T1','a2/a1','p02/p01'), key = "no1")
z0 = (g5-1)/g5
if s5 == "M1":
    x = st.number_input('Enter', min_value = 1.001)
    noshow(x,g5)
elif s5 == "M2":
    x = st.number_input('Enter', max_value = 1.0, min_value = z0)
    x = float(x)
    def f(z):
        M = z
        f0 = noM2(M,g5) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    noshow(M,g5)
elif s5 == "P2/P1":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = noP_ratio(M,g5) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    noshow(M,g5)
elif s5 == "rho2/rho1":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = nod_ratio(M,g5) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    noshow(M,g5)
elif s5 == "T2/T1":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = noT_ratio(M,g5) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    noshow(M,g5)
elif s5 == "a2/a1":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = noa_ratio(M,g5) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    noshow(M,g5)
elif s5 == "p02/p01":
    x = st.number_input('Enter', max_value = 1.0, min_value = 0.001)
    def f(z):
        M = z
        f0 = noPs_ratio(M,g5) - x
        return f0
    zGuess = 5
    z = fsolve(f,zGuess)
    M = float(z)
    noshow(M,g5)

#RAYLEIGH FLOW

def raTs_ratio(M,g):
    return (2*(g+1)*(1 + ((g-1)/2)*M*M)*M*M)/((1 + g*M*M)**2)
def raT_ratio(M,g):
    return (((g+1)*M)/(1 + g*M*M))**2
def raP_ratio(M,g):
    return (g+1)/(1 + g*M*M)
def raPs_ratio(M,g):
    return (((g+1)/(2*(1 + ((g-1)/2)*M*M)))**(g/(1-g)))*(g+1)/(1 + g*M*M)
def rad_ratio(M,g):
    return (g + 1)*M*M/(1 + g*M*M)
st.subheader("Rayleigh Flow")
g2 = st.number_input('The value of gamma is ', min_value = 1.001, key = "ra")
def rashow(M,g):
    st.write("M = " + str(round(abs(M),4)))
    st.write("T/T* = " + str(round(raT_ratio(M,g2),4))) 
    st.write("T0/T0* = " + str(round(raTs_ratio(M,g2),4)))
    st.write("p/p* = " + str(round(raP_ratio(M,g2),4)))
    st.write("p0/p0* = " + str(round(raPs_ratio(M,g2),4))) 
    st.write("rho*/rho = " + str(round(rad_ratio(M,g2),4)))
s2 = st.selectbox('Select the variable',('M', 'T/T*', 'T0/T0*','rho*/rho','p/p*','p0/p0*'), key = "ra1")
if s2 == "M":
    x = st.number_input('Enter', min_value = 0.0001)
    rashow(x,g2)
elif s2 == "T/T*":
    x = st.number_input('Enter')
    x = float(x)
    def f(z):
        M = z
        f0 = raT_ratio(M,g2) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("First solution:")
    rashow(M,g2)
    def f(z):
        M = z
        f0 = raT_ratio(M,g2) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Second solution:")
    rashow(M,g2)
elif s2 == "T0/T0*":
    x = st.number_input('Enter', max_value = 1.0)
    x = float(x)
    def f(z):
        M = z
        f0 = raTs_ratio(M,g2) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    rashow(M,g2)
    def f(z):
        M = z
        f0 = raTs_ratio(M,g2) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    rashow(M,g2)
elif s2 == "p/p*":
    x = st.number_input('Enter')
    def f(z):
        M = z
        f0 = raP_ratio(M,g2) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    rashow(M,g2)
elif s2 == "p0/p0*":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = raPs_ratio(M,g2) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    rashow(M,g2)
    def f(z):
        M = z
        f0 = raPs_ratio(M,g2) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    rashow(M,g2)
elif s2 == "rho*/rho":
    x = st.number_input('Enter')
    def f(z):
        M = z
        f0 = rad_ratio(M,g2) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    rashow(M,g2)

#FANNO FLOW

st.subheader("Fanno Flow")
from math import log
def fric(M,g):
    return (1 - M*M)/(g*M*M) + ((g+1)/(2*g))*log((g+1)*M*M/(2*(1+((g-1)/2)*M*M)))
def faP_ratio(M,g):
    return (1/M)*(((g+1)/(2*(1 + ((g-1)/2)*M*M)))**(0.5))
def faT_ratio(M,g):
    return (g+1)/(2*(1 + ((g-1)/2)*M*M))
def faPs_ratio(M,g):
    return (1/M)*((faT_ratio(M,g))**((g+1)/(2*(1-g))))
def faF_ratio(M,g):
    return (1 + g*M*M)/(M*((2*(g+1)*(1 + ((g-1)/2)*M*M))**(0.5)))
def fad_ratio(M,g):
    return M*((faT_ratio(M,g))**(0.5))
g3 = st.number_input('The value of gamma is ', min_value = 1.001, key = "fa")
def fashow(M,g):
    st.write("M = " + str(round(abs(M),4)))
    st.write("T/T* = " + str(round(faT_ratio(M,g3),4))) 
    st.write("p/p* = " + str(round(faP_ratio(M,g3),4)))
    st.write("p0/p0* = " + str(round(faPs_ratio(M,g3),4))) 
    st.write("rho*/rho = " + str(round(fad_ratio(M,g3),4)))
    st.write("F/F* = " + str(round(faF_ratio(M,g3),4)))
    st.write("4fL/D = " + str(round(fric(M,g3),4)))
s3 = st.selectbox('Select the variable',('M', 'T/T*', 'p/p*','rho*/rho','F/F*','p0/p0*', '4fL/D'), key = "fa1")
if s3 == "M":
    x = st.number_input('Enter', min_value = 0.0001, key = "fa2")
    fashow(x,g3)
elif s3 == "T/T*":
    x = st.number_input('Enter')
    x = float(x)
    def f(z):
        M = z
        f0 = faT_ratio(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    fashow(M,g3)
elif s3 == "p0/p0*":
    x = st.number_input('Enter', min_value = 1.0)   
    x = float(x)
    def f(z):
        M = z
        f0 = faPs_ratio(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    fashow(M,g3)
    def f(z):
        M = z
        f0 = faPs_ratio(M,g3) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    fashow(M,g3)
elif s3 == "p/p*":
    x = st.number_input('Enter')
    def f(z):
        M = z
        f0 = faP_ratio(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    fashow(M,g3)
elif s3 == "F/F*":
    x = st.number_input('Enter', min_value = 1.0)
    def f(z):
        M = z
        f0 = faF_ratio(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    fashow(M,g3)
    def f(z):
        M = z
        f0 = faF_ratio(M,g3) - x
        return f0
    zGuess = 5
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    fashow(M,g3)
elif s3 == "rho*/rho":
    x = st.number_input('Enter')
    def f(z):
        M = z
        f0 = fad_ratio(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    fashow(M,g3)
elif s3 == "4fL/D":
    x = st.number_input('Enter')
    def f(z):
        M = z
        f0 = fric(M,g3) - x
        return f0
    zGuess = 0.1
    M = fsolve(f,zGuess)
    M = float(M)
    st.write("Subsonic solution:")
    fashow(M,g3)
    def f(z):
        M = z
        f0 = fric(M,g3) - x
        return f0
    zGuess = 10
    M = fsolve(f,zGuess)
    M = float(M)
    st.write(" ")
    st.write("Supersonic solution:")
    fashow(M,g3)

#OBLIQUE SHOCK  

st.subheader("Oblique Shock")
from math import sin
from math import tan
from math import cos
from math import atan
import math
def obT(M,B,g):
    return atan((2/tan(B))*(M*M*sin(B)*sin(B) - 1)/(M*M*(g + cos(2*B)) + 2))
import numpy as np
from scipy.optimize import fsolve
def obbeta1(M,g,x):   
    x = x*(math.pi)/180
    def B(z):
        f0 = obT(M,z,g) - x
        return f0
    zGuess = 0.01
    b = fsolve(B,zGuess)
    b = float(b)
    b *= 180/(math.pi)
    return b
def obbeta2(M,g,x):
    x = x*(math.pi)/180
    def B2(z):
        f0 = obT(M,z,g) - x
        return f0
    zGuess = 1.57
    b = fsolve(B2,zGuess)
    b = float(b)
    b *= 180/(math.pi)
    return b
def obP_ratio(M,B,g):
    B *= (math.pi)/180
    return 1 + (2*g/(g+1))*(M*M*sin(B)*sin(B) - 1)
def obMn(M,B,g):
    B *= (math.pi)/180
    return ((M*M*sin(B)*sin(B) + 2/(g-1))/((2*g/(g-1))*M*M*sin(B)*sin(B) - 1))**(0.5)
g4 = st.number_input('The value of gamma is ', min_value = 1.001, key = "ob")
M1 = st.number_input('M1 = ', min_value = 1.001, key = "ob2")
def obshow1(M,T,g):
    B1 = obbeta1(M,g,T)
    B2 = obbeta2(M,g,T)
    M2_1 = obMn(M,B1,g)/sin((B1-T)*(math.pi)/180)
    M2_2 = obMn(M,B2,g)/sin((B2-T)*(math.pi)/180)
    M2_1 = abs(M2_1)
    M2_2 = abs(M2_2)
    st.write("M1 = " + str(round(M1,2)))
    st.write("Turn angle (theta) = " + str(round(T,1)))
    st.write("")
    st.write("Weak Solution :-")
    st.write("Wave angle (beta) = " + str(round(B1,2)))
    st.write("p2/p1 = " + str(round(obP_ratio(M,B1,g), 3)))
    st.write("M2 = " + str(round(M2_1,3)))
    st.write("")
    st.write("Strong Solution :-")
    st.write("Wave angle (beta) = " + str(round(B2,2)))
    st.write("p2/p1 = " + str(round(obP_ratio(M,B2,g), 3)))
    st.write("M2 = " + str(round(M2_2,3)))
def obshow2(M,T,g,B):
    M2 = obMn(M,B,g)/sin((B-T)*(math.pi)/180)
    M2 = abs(M2)
    st.write("M1 = " + str(round(M1,2)))
    st.write("Wave angle (beta) = " + str(round(B,2)))
    st.write("p2/p1 = " + str(round(obP_ratio(M,B,g), 3)))
    st.write("M2 = " + str(round(M2,3)))
s4 = st.selectbox('Select the variable',('Turn angle (theta)', 'Wave angle (Beta)'), key = "ob1")
if (s4 == "Turn angle (theta)"):
    x = st.number_input('Enter ', key = "ob3")
    x = float(x)
    obshow1(M1,x,g4)
else:
    B = st.number_input('Enter ', key = "ob4")
    T = obT(M1,B,g4)
    obshow2(M1,T,g4,B)

st.write("")
st.write("")
st.write("")
st.write("Made by Abhyudaya Nerwat")
st.write("under Prof. Ankit Bansal")
