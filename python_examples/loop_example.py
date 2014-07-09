import numpy as np

# Loop over a list.
mylist = ['a','b','c','d']

for val in mylist:
    print val


print "\nChange the variable to anything you like!"
for x in mylist:
    print "this is in the loop"
    print x
    print "This is also in the loop!"

print "This is not in the loop, because it is not indented anymore."

other_mylist = [1,2,3,4,5,6,7,9]

for x in mylist:
    for y in other_mylist:
        print "%s %d" % (x,y)


# This also works for arrays.
values = np.arange(0,10,0.1)

'''
for n in values:
    print n
'''

# You can also do things C-like.
num = len(values)

'''
for i in range(0,num):
    print values[i]
'''

# You can access the last element of an array, with -1
print "\n\n"
print values[-1]

# You can access the next-to-last element of an array, with -2
print "\n\n"
print values[-2]
