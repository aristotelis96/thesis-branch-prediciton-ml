import numpy as np
import matplotlib.pyplot as plt
import os



averageIPC = {"TAGE8":[], "TAGE64":[], "PB": [], "Perfect_H2P_TAGE64":[]}

benchs = ['600.perlbench_s', '605.mcf_s', '620.omnetpp_s', '623.xalancbmk_s', '625.x264_s', '631.deepsjeng_s', '641.leela_s', '648.exchange2_s', '657.xz_s']

branches = ["TAGE8", "TAGE64", "Perfect_H2P_TAGE64", "PB"]

#get Weights for each simpoint
weight_DIR="C:/Users/Aristotelis/Desktop/diploma/weights-and-simpoints-speccpu"
weights=dict()
for bench in benchs:
    w = open(weight_DIR+"/"+bench+"/weights.out")
    sim = open(weight_DIR+"/"+bench+"/simpoints.out")
    weights[bench]=[]
    for line in w:
        if (float(line)<0.01):
            sim.readline()
            continue
        weights[bench].append([sim.readline()[:-1], float(line)])

#calculate for each branch
base={}
for branch in branches:
    scales=os.listdir("./scalingResults/"+branch)    
    #for each scale (x01, x02, x04 etc)    
    for scale in scales:    
        totalAverage=1
        for bench in benchs:
            # each bench has several weights/simpoints
            for weight in weights[bench]:
                benchFile = bench+"-"+weight[0]+"B.champsimtrace.xz._."+scale+".txt"                
                with open("./scalingResults/"+branch+"/"+scale+"/"+benchFile) as f:
                    numerator = 0 
                    denominator = 0
                    for line in f:
                        if line.startswith("CPU 0 cumulative IPC:"):
                            line = line.split()
                            numerator+=float(line[4])*weight[1]
                            denominator+=weight[1]    
            #totalAverage=numerator/denominator
            if(scale=="x01_tage8"):
                base[bench] = numerator/denominator
                totalAverage = 1
            else:
               totalAverage *= (numerator/denominator)/base[bench]
        totalAverage=totalAverage**(1/float(len(benchs)))
        print(branch, totalAverage)
        averageIPC[branch].append(totalAverage)
print(averageIPC)
print(base)
ind = np.arange(len(scales))
width = 0.35

p4 = plt.bar(ind, averageIPC["PB"], width, color='blue')
p3 = plt.bar(np.arange(4), averageIPC["Perfect_H2P_TAGE64"], width, color='lime')
p2 = plt.bar(ind, averageIPC["TAGE64"], width, color='orange')
p1 = plt.bar(ind, averageIPC["TAGE8"], width, color='silver')

plt.legend([p4,p3,p2,p1],["PB","Perfect_H2P_TAGE64","TAGE-SC-L-64KB","TAGE-SC-L-8KB"])


plt.xticks(ind, [i[0:3] for i in scales])
plt.yticks(np.arange(1,2.5,0.25))
plt.ylim(1, 2.5)
ax = plt.axes()
ax.yaxis.grid(True)
plt.title("Average for all benchmarks")
plt.ylabel("IPC relative to x01-TAGE-8KB")
plt.xlabel("Pipeline capacity scaling")
plt.savefig("./graphs/all_average.png")
#plt.show()
plt.clf() #clear
#exit()
#plot for each benchmark
base={}
for bench in benchs:
    IPC = {"PB":[], "TAGE8":[], "Perfect_H2P_TAGE64":[], "TAGE64":[]}
    for branch in branches:
        scales=os.listdir("./scalingResults/"+branch)
        for scale in scales:
            numerator = 0 
            denominator = 0
            for weight in weights[bench]:
                benchFile = bench+"-"+weight[0]+"B.champsimtrace.xz._."+scale+".txt"                
                with open("./scalingResults/"+branch+"/"+scale+"/"+benchFile) as f:
                    for line in f:
                        if line.startswith("CPU 0 cumulative IPC:"):
                            line = line.split()
                            numerator+=float(line[4])*weight[1]
                            denominator+=weight[1]
            if(scale=="x01_tage8"):
                base[bench] = numerator/denominator
                IPC[branch].append(1)
            else:
               IPC[branch].append(numerator/(denominator*base[bench]))
            #IPC[branch].append(numerator/denominator)
    plt.clf() #clear

    ind = np.arange(len(scales))
    width = 0.35

    p4 = plt.bar(ind, IPC["PB"], width, color='blue')
    p3 = plt.bar(np.arange(4), IPC["Perfect_H2P_TAGE64"], width, color='lime')
    p2 = plt.bar(ind, IPC["TAGE64"], width, color='orange')
    p1 = plt.bar(ind, IPC["TAGE8"], width, color='silver')
    plt.legend([p4,p3,p2,p1],["Perfect Branch Prediction","Perfect_H2P_TAGE64","TAGE-SC-L-64KB","TAGE-SC-L-8KB"])

    plt.xticks(ind, [i[0:3] for i in scales])
    plt.yticks(np.linspace(1, max(IPC["TAGE8"]+IPC["PB"]+IPC["TAGE64"]+IPC["Perfect_H2P_TAGE64"])+0.2,10))
    plt.ylim(1, max(IPC["PB"]+IPC["TAGE64"]+IPC["TAGE8"]+IPC["Perfect_H2P_TAGE64"])+0.2)
    ax = plt.axes()
    ax.yaxis.grid(True)
    plt.title(bench)
    plt.ylabel("IPC relative to x01-TAGE-8KB")
    plt.xlabel("Pipeline capacity scaling")
    plt.savefig("./graphs/"+bench.replace(".txt", "")+".png")
    #plt.show()
