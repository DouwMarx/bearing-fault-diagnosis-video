import sympy as sp

# model the sytems of equations
# r_c*omega_c = r_s*omega_s + r_p*omega_p
# r_R*omega_R = r_c*omega_c + r_p*omega_p
# r_c = r_s + r_p
# r_R = r_c + r_p

# Define the symbols
r_c, r_s, r_p, r_R = sp.symbols('r_c r_s r_p r_R')
omega_c, omega_s, omega_p, omega_R = sp.symbols('omega_c omega_s omega_p omega_R')

# Define the equations
eq1 = sp.Eq(r_c*omega_c, r_s*omega_s + r_p*omega_p)
eq2 = sp.Eq(r_R*omega_R, r_c*omega_c + r_p*omega_p)
eq3 = sp.Eq(r_c, r_s + r_p)
eq4 = sp.Eq(r_R, r_c + r_p)

# omega_s,omega_R and omega_c are known constants

# Solve the equations for omega_p in terms of omega_s and omega_c
sol = sp.solve([eq1,eq2,eq3,eq4],[omega_p,r_c,r_s,r_R])
# print(sol)



# Define a new set of equations

eqA = sp.Eq(omega_c/omega_s, r_s/(2*r_c))
eqB = sp.Eq(omega_p*r_p, omega_c*r_c - omega_s*r_s)
eqC = sp.Eq(r_c, r_s + r_p)

# Solve the equations for omega_p
sol = sp.solve([eqA,eqB,eqC],[omega_p,r_c,r_s])
# sol = sp.solve([eqA,eqB,eqC],[omega_p,r_c,r_s,r_p])
print(sol)
