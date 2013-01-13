 from pulp import *

# define problem
prob = LpProblem("Weighted Set Cover Problem", LpMinimize)

# define variables and ranges 0 <= xi <= 1
x1 = LpVariable("set1", 0,1)
x2 = LpVariable("set2", 0,1)
x3 = LpVariable("set3", 0,1)
x4 = LpVariable("set4", 0,1)
x5 = LpVariable("set5", 0,1)

#define objective function first
prob += x1 + x2 + x3 + x4 + x5, "The Total Number of sets used"

#define constraints
prob += 1.0*x1 + 1.0*x2 + 0.0*x3 + 1.0*x4 + 0.0*x5 >=1, "elem1Requirment"
prob += 0.0*x1 + 1.0*x2 + 1.0*x3 + 0.0*x4 + 0.0*x5 >=1, "elem2Requirment"
prob += 1.0*x1 + 0.0*x2 + 1.0*x3 + 0.0*x4 + 1.0*x5 >=1, "elem3Requirment"
prob += 0.0*x1 + 1.0*x2 + 1.0*x3 + 0.0*x4 + 1.0*x5 >=1, "elem4Requirment"
prob += 0.0*x1 + 0.0*x2 + 1.0*x3 + 1.0*x4 + 0.0*x5 >=1, "elem5Requirment"

# The problem is solved using PuLP's choice of solver
prob.solve()
print "Status:", LpStatus[prob.status], ""
for variable in prob.variables():
    print variable.name, "=", variable.varValue

# The optimised objective function value is printed to the screen
print "Total Cost = ", value(prob.objective)


