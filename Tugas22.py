# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:51:37 2019

@author: I WAYAN SUMIARTA
"""
import pandas as pd
import numpy as np

tabel = pd.read_csv('D:/AI/data_latih_opsi_2.csv', names=["Suhu","Langit","Waktu","Kelembaban","Hasil"], header=1)

individu = 10
numparents = 2
tabel = tabel.values
data = tabel[:,:]

x = lambda f, i: (f[i] / sum(f))
def roulette(fit):
    r = np.random.uniform(0,1)
    indv = 0
    while r > 0:
        r -= x(fit,indv)    
        indv += 1
    return indv-1

def fitness(dataset,rules):
    bn = 0
    t = len(dataset)
    for i, v in enumerate(dataset):
        has = tree(v[:-1], rules)
        if v[-1] == has:
            bn += 1
        elif has == "unseen":
            t -= 1
    return bn
    return t

def encode(decoded):
    bit1, bit4 =  np.zeros(3)
    bit2, bit3 = np.zeros(4)
    bit5 = np.zeros(1)
    bit1[decoded[0]],bit2[decoded[1]],bit3[decoded[2]],bit4[decoded[3]] = 1
    bit5[0] = decoded[4]
    return np.concatenate([bit1, bit2, bit3, bit4, bit5], axis=0)

def decode(encoded):
    bits = [None] * 5
    bits[0], bits[1], bits[2], bits[3], bits[4] = encoded[:3], encoded[3:7], encoded[7:11], encoded[11:14], encoded[-1]
    nbits = []
    for i, v in enumerate(bits[:-1]):
        bitz = []
        for j, k in enumerate(v):
            if k == 1:
                bitz.append(j)
        nbits.append(bitz)
    nbits.append([bits[-1]])
    return nbits

def initializeRule():
    krm = [None] * 5
    krm[0],krm[3] = np.zeros(3)
    krm[1], krm[2] = np.zeros(4)
    krm[4] =  np.zeros(1)
    
    for i, v in enumerate(krm[:-1]):
        while sum(krm[i]) == 0:
            for j, k in enumerate(v):
                krm[i][j] = int(np.random.rand() > .10)
    krm[4] = np.array([int(np.random.rand() > .5)])
    nkrm = np.concatenate(krm)
    return nkrm
def initialize(individu, prule):
    population = []
    for i in range(individu):
        rules = []
        for j in range(prule):
            rules.append(initializeRule())
        population.append(np.concatenate(rules, axis=0))
    return population

def mutation(individu):
    if (individu != []):
        probability = np.random.randint(0,100)
        if (probability == 1):
            num1 = np.random.randint(0,len(individu[0])-1)
            num2 = np.random.randint(0,len(individu[1])-1)
            if individu[0][num1] == 0:
                individu[0][num1] = 1
            else:
                individu[0][num1] = 0
            if individu[1][num2] == 0:
                individu[1][num2] = 1
            else:
                individu[1][num2] = 0
    return individu
def crossover(parents):
    probability = np.random.randint(0,100)
    individu = []
    if (probability <= 70):
        infoChild = {}
        point = np.random.randint(1,len(parents[0]))
        genidv = np.append([parents[0][:point]],[parents[1][point:]])
        infoidv = {
            genidv,
        }
        individu.append(infoidv)
        genidv2 = np.append([parents[1][:point]],[parents[0][point:]])
        infoidv = {
            genidv2,
        }
        individu.append(infoChild)
    return individu

def tree(test, rules):
    if (test != []):
        rules.sort(key=lambda rules:rules['fitness'],reverse=False)
        size = len(test[0]['genotype'])
        for i in range (len(test)):
            x1 = encode(test[i]['genotype'][:size//2], -3, 3)
            x2 = encode(test[i]['genotype'][size//2:], -2, 2)
            test[i]['x1'] = x1
            test[i]['x2'] = x2
            test[i]['f'] = (x1,x2)
            test[i]['fitness'] = fitness((x1,x2))
        rules.remove(rules[0])
        rules.remove(rules[0])
        rules.append(test[0])
        rules.append(test[1])
    return rules


generasi = 50

indvs = np.array(initialize(individu,60))

for gen in range(generasi):
    fitn = []
    for i, v in enumerate(indvs):
        fitn.append(fitness(data, v))

    parents = []
    for i in range(numparents):
        parents.append(np.array(indvs[roulette(fitn)]))
    parents = np.array(parents)

    offsprings = np.array(crossover(parents))
    mutated_offspring = []
    for i, offspring in enumerate(offsprings):
        mutated_offspring.append(mutation(offspring))
    mutated_offspring = np.array(mutated_offspring)

    indvs[0:parents.shape[0]] = parents
    indvs[-parents.shape[0]:] = mutated_offspring

    n_fitn = [fitness(data, i) for i in indvs]
    best = np.max(n_fitn)
    idx = np.argmax([fitness(data, i) for i in indvs],axis=0)

    print(best,"at index:", idx)