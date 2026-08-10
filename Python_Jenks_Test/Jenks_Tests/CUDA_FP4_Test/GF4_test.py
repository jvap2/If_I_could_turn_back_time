import random

gf4v= [0,0.08,0.174,0.283,0.395,0.525,0.696,1]

def center(values):
    #return center point
    boundaries =[0]
    for i in range(0,len(values)-1):
        boundaries.append((values[i]+values[i+1])/2)
    return boundaries
        
    
                   

def quant(values,boundaries,mean, sd, num):
    # values is a list of positive quantized values
    # boundaries is a list of the thresholds inactual values
    #  that are the borders for the quantized values
    # mean and sd are the statistics for the gaussian distro of num values
    # The result is a list of counts of quantized values
    # Thus is a random x is between boundaries[j] and boundaries [j+1]
    # Then x sould be represented as values[j]
    counts=[0]*len(values)
    boundaries.append(1e6*boundaries[-1])
    sum = 0
    for i in range(0,num):
        x=random.gauss(mean,sd)
        sum += x
        x=abs(x)
        for j in range(0,len(values)):
            if (boundaries[j]<=x)and(x<boundaries[j+1]):
                counts[j]+=1
                exit
    mean1 = sum/num
    print("The values had mean",mean1)
    print(counts)


def testgauss(m,sd,n):
    # generate n samples from gauss and see if statistics match
    # Just a validation that random.gauss produces gaussian distrobution 
    mean = 0
    std = 0
    for i in range(0,n):
        x= random.gauss(m,sd)
        if x<0: continue
        mean += x
        std += (x-m)**2
        #print(x,(x-m)**2,std)
    mean=mean/n
    stdev= (std/n)**0.5
    print("input",m,sd)
    print("stats",mean,stdev)

def find_edge(target,m,sd,num,init,window):
    # return threshold that returns num1 counts
    # plus or minus window
    # from num samples with statistics m and sd from gauss
    # with initial edge init
    flag = True
    edge = m+4*sd
    while flag:
        count = 0
        for i in range(0,num):
            x= abs(random.gauss(m,sd))
            if x<=edge: count +=1
        diff= count-target
        #print(edge,count,diff)
        if abs(diff)<window:
            flag = False
        else: edge += -edge*diff/num
    print("Got",count,"with edge",edge)
    return edge

#test that random.gauss is gaussian
# or at least retuens correct mean, std
#testgauss(0,.25,1000000)

print("Now compute exact boundaries for equal probability mass intervals")  
exact = [0]*(len(gf4v)+1)
exact[1]=find_edge(10000,0,0.25,80000,1,10)
exact[2]=find_edge(20000,0,0.25,80000,1,10)
exact[3]=find_edge(30000,0,0.25,80000,1,10)
exact[4]=find_edge(40000,0,0.25,80000,1,10)
exact[5]=find_edge(50000,0,0.25,80000,1,10)
exact[6]=find_edge(60000,0,0.25,80000,1,10)
exact[7]=find_edge(70000,0,0.25,80000,1,10)
exact[8]=10
print("Exact boundaries:",exact)
print("GF4 values:",gf4v)
   
gf4be=center(gf4v)
print("Middle spaced boundaries:",gf4be)

print("Try center boundaries")
quant(gf4v,gf4be,0,0.25,80000)


print("Try computed boundaries")
quant(gf4v,exact,0.0,0.25,80000)
