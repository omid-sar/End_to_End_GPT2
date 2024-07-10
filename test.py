
import numpy
my_list = [1, 2, 3, 4]
list_2 = list(map(lambda x: x*x, my_list))
print(list_2)
list_3 = [i**2 for i in my_list]
print(list_3)
shap(my_list)
