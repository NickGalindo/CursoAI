from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Any, Dict, Optional    

# DATACLASSES

@dataclass
class Country:
    name: str
    size: float = 0.0
    

@dataclass()
class Passport:
    number: str
    issued_by_country: Country

@dataclass()
class Person():
    name: str
    nationality: Country
    passport: Optional[Passport] = None
    can_travel: bool = field(init=False)

    def __post_init__(self):
        self.can_travel = bool(self.passport)
        return self.can_travel

# Creation 

colombia = Country(name='Colombia', size=1142.00)
venezuela = Country(name='Venezuela')
peru = Country('Perú',20)

print(peru)
print(f'Colombia information: {str(colombia)} \n')

first_passport = Passport(number='1298348756', issued_by_country=colombia)

colombian_person = Person(name='Luis García', nationality=colombia, passport=first_passport)

print(colombian_person)

print(colombian_person.__post_init__())


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