
# mon = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# score = [5, 4, 3, 2]

# def compute_mon_score(mon_num):
#     if mon_num == 5:
#         score = 0.5
#     if mon_num == 4:
#         score ==0.8
#     if mon_num == 3:
#         score ==1.9
#     if

score_list = list()
for mon_1 in [5, 4, 3, 2]:
    for mon_2 in [5, 4, 3, 2]:
        for mon_3 in [5, 4, 3, 2]:
            for mon_4 in [5, 4, 3, 2]:
                for mon_5 in [5, 4, 3, 2]:
                    for mon_6 in [5, 4, 3, 2]:
                        for mon_7 in [5, 4, 3, 2]:
                            for mon_8 in [5, 4, 3, 2]:
                                for mon_9 in [5, 4, 3, 2]:
                                    for mon_10 in [5, 4, 3, 2]:
                                        for mon_11 in [5, 4, 3, 2]:
                                            for mon_12 in [5, 4, 3, 2]:
                                                score_tmp = (mon_1+mon_2+mon_3+mon_4+mon_5+mon_6+mon_7+mon_8+mon_9+mon_10+mon_11+mon_12)/12
                                                score_list.append(round(score_tmp, 1))

lengh = len(score_list)
print(len(score_list))
print(score_list[:10])

list_a_plus = list()
list_a = list()
list_b = list()
list_c = list()

for i in score_list:
    if 4.5 <= i <= 5.0:
        list_a_plus.append(i)
    elif 3.5 <= i <= 4.4:
        list_a.append(i)
    elif 2.5 <= i <= 3.4:
        list_b.append(i)
    else:
        list_c.append(i)

print(len(list_a_plus)/lengh)
print(len(list_a)/lengh)
print(len(list_b)/lengh)
print(len(list_c)/lengh)












