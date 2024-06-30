def add(x,y):
    sum = x+y
    return sum

X=2

class car:

    def __init__(self, model):
        self.model = model
    def eva(self, year: int=2024, pre_owned=False):
        if self.model=="Toyota" and year > 2018 and pre_owned == False :
            condition = "Great"
        elif self.model=="Toyota" and year < 2018 and pre_owned == False:
            condition = "Good"
        elif self.model=="Toyota" and year < 2018 :
            condition = "Fair"
        else:
            condition = "Dont waste your money"
        print(f"This {self.model} made in {year} and pre_owned:{pre_owned} \n CONDITION: {condition}")
        return condition


car1 = car("Toyota")
car2 = car("")
car3 = car("BMW")
car1.eva(year=2012)
car2.eva(year=2022, pre_owned=True)
car3.eva(year=2020)

car3.eva()

        
if __name__ == "__main__":
    x = input("Num 1: ")
    y = input("Num 2: ")
    import pdb; pdb.set_trace()
    z = add(x,y)
    print(z)
