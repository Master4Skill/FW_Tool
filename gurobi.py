import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import matplotlib.pyplot as plt


def plot_results(df_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red", "green", "blue"]

    # Extend each series with the last value
    x_values = df_results.index.append(pd.Index([df_results.index.max() + 1]))

    ax.step(
        x_values,
        pd.concat([df_results["x"], pd.Series(df_results["x"].iloc[-1])])
        - pd.concat([df_results["y"], pd.Series(df_results["y"].iloc[-1])]),
        where="post",
        label="x - y",
        color=colors[1],
    )
    ax.step(
        x_values,
        pd.concat([df_results["x"], pd.Series(df_results["x"].iloc[-1])])
        + pd.concat([df_results["z"], pd.Series(df_results["z"].iloc[-1])]),
        where="post",
        label="x + z",
        color=colors[2],
    )
    ax.step(
        x_values,
        pd.concat([df_results["x"], pd.Series(df_results["x"].iloc[-1])]),
        where="post",
        label="x",
        color=colors[0],
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Optimization Results")
    ax.legend()
    plt.show()


"""
m = gp.Model("mip1")

# Create variables
x = m.addVar(vtype=gp.GRB.INTEGER, name="x")
y = m.addVar(vtype=gp.GRB.INTEGER, name="y")
z = m.addVar(vtype=gp.GRB.INTEGER, name="z")

# Set objective
m.setObjective(x + y + 2 * z, gp.GRB.MAXIMIZE)

# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(5 * x + 2 * y + 3 * z <= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")

m.optimize()
print("Obj: %g" % m.objVal)
print(x.X)
print(y.X)
print(z.X)
"""


df_results = pd.DataFrame()
P_to_dem = [8, 14, 5, 16, 7, 9, 10, 11, 6, 12]
I = list(range(len(P_to_dem)))

m = gp.Model("Winco Page 232 Winston")

x = m.addVars(I, vtype=gp.GRB.CONTINUOUS, name="x")
y = m.addVars(I, vtype=gp.GRB.CONTINUOUS, name="y")
z = m.addVars(I, vtype=gp.GRB.CONTINUOUS, name="z")
f = m.addVar(name="f")
X = m.addVar(name="X")
E_stored = m.addVars(I, vtype=gp.GRB.CONTINUOUS, name="E_stored")

m.addConstr(f == 1 * X)
m.setObjective(f, gp.GRB.MINIMIZE)
m.addConstr(X == gp.quicksum(x[i] for i in I))

for i in range(len(P_to_dem)):
    P = P_to_dem[i]
    m.addConstr(x[i] + z[i] == y[i] + P, name=f"c1_{i}")
    m.addConstr(x[i] <= 12, name=f"c2_{i}")
    m.addConstr(x[i] >= 8, name=f"c3_{i}")
    if i == 0:  # for the first period, E_stored_prev is 0
        m.addConstr(E_stored[i] == y[i] - z[i], name=f"c4_{i}")
    else:  # for other periods, E_stored_prev is E_stored from last period
        m.addConstr(E_stored[i] == E_stored[i - 1] + y[i] - z[i], name=f"c4_{i}")
    m.addConstr(E_stored[i] <= 10, name=f"c5_{i}")

m.setParam("OutputFlag", 0)
m.optimize()

x_m = m.getAttr("x", x)
y_m = m.getAttr("x", y)
z_m = m.getAttr("x", z)
E_stored_m = m.getAttr("x", E_stored)

for key in x_m.keys():
    data = {
        "x": x_m[key],
        "y": y_m[key],
        "z": z_m[key],
        "E_stored": E_stored_m[key],
        "f": m.objVal,
        "P_to_dem": P_to_dem[key],
    }
    df_results = pd.concat(
        [df_results, pd.DataFrame(data, index=[0])], ignore_index=True
    )

print(df_results)
plot_results(df_results)
