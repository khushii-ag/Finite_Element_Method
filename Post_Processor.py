import numpy as np
import Super_Processor as pro
import csv

prim_var, gcv = pro.super_processor()

with open('answer2.csv', "w", newline="") as f:
    writer = csv.writer(f)
    list = ["X coordinate"]
    list2 = ["Y coordinate"]
    list3 = ["Primary Variable"]
    
    for i in range(len(gcv)):
        list.append(gcv[i][0])
        list2.append(gcv[i][1])
        list3.append(prim_var[i][0])
    writer.writerow(list)
    writer.writerow(list2)
    writer.writerow(list3)



