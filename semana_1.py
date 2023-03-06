#import numpy as np
from pprint import pprint
from typing import List, Any, Dict

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# DATACLASSES
@dataclass
class Hamburger:
    bun_type: str
    bun_toasted: bool
    tomatoes: bool
    cheese: bool
    meat: str
    onion: bool

@dataclass
class GarlicBread:
    toasted: bool

@dataclass
class Order:
    order_num: int
    items: List[Any] = field(default_factory = lambda : [GarlicBread(toasted=True)])
    
    def __post_init__(self):
        print(f"initializing order num: {self.order_num}")
        return 


# GENERADORES
def fibGenerator(n: int):
    fib_dp: List[int] = [0, 1, 0]
    if n < 2:
        return fib_dp[n]
    i = 2
    while i <= n:
        fib_dp[i%3] = fib_dp[(i-1)%3] + fib_dp[(i-2)%3]
        yield fib_dp[i%3]
        i += 1

for i in fibGenerator(10):
    print(f"- {i}")

for i in enumerate(fibGenerator(10)):
    print(i)

# DataFrames
data: Dict[str, List] = {
    "nombre": ["Nicolas", "Mateo", "Sebastian", "Eric", "Andres", "Jorge"],
    "apellido": ["Galindo", "Gutierrez", "DeLaFonte", "Jara", "Garcia", "Paez"]
}

data_df = pd.DataFrame(data)

for row in data_df.iterrows():
    pprint(row)

# LOAD DATA
electric_motor_data = pd.read_csv("electric_motor_temperature.csv")

pprint(electric_motor_data.columns)

print(electric_motor_data.describe())

electric_motor_data = electric_motor_data[electric_motor_data["u_q"] > 0]

print(electric_motor_data[:10])

print(electric_motor_data["u_q"].value_counts(dropna=False))
print(electric_motor_data.columns)

print(electric_motor_data.groupby(["u_q", "coolant"]).sum())

# PLOTTING DATA
sub_electric_motor_data = electric_motor_data.groupby("coolant")["torque"].mean().reset_index()

print(sub_electric_motor_data)

x = sub_electric_motor_data["coolant"].to_numpy()
y = sub_electric_motor_data["torque"].to_numpy()

plt.plot(x, y, label="avg_torque_to_coolant")
plt.show()

sin_x = np.sin(x)

plt.plot(x, sin_x, label="sin_x")
plt.title("Electric Motors Analysis")
plt.legend()
plt.show()


# PYTORCH
data = [[[42, 69], [86, 77]], [[99, 420], [0, 1]]]
t_data = torch.tensor(data)

data = np.array(data)
t_data = torch.tensor(data)

print(t_data.shape)
print(t_data.dtype)
print(t_data.device)

if torch.cuda.is_available():
    t_data = t_data.to("cuda:1")
    print(t_data.device)

t_new = torch.zeros(data.shape)
t_new[:,0] = 1

t_data = t_data.double()

print(t_data * t_new)
print(t_data @ t_new.mT.double())
