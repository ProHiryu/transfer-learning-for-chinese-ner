#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - songheqi <songheqi1996@gmail.com>

with open('data/transfer_train','r') as f:
    train_lines = f.readlines()

with open('data/transfer_test','r') as f:
    test_lines = f.readlines()

print(len(train_lines))
count = 0
# with open('data/transfer_train.1','w+') as f:
#     for l in train_lines:
#         f.write(l)
#         if l == '\n':
#             count += 1
#         if count == 800:
#             break
            
# print(count)
# count = 0
# with open('data/transfer_test.1','w+') as f:
#     for l in test_lines:   
#         print(l)
#         f.write(l)
#         if l == '\n':
#             count += 1
#         if count == 250:
#             break
    
# print(count)