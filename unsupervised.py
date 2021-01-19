import gym
import math
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
from multiprocessing import*
env = gym.make("CartPole-v1")
observation = env.reset()
networks = []
population = 5
generations = 10
def modeltochromosome(model):
    chromosome = []

    for layer in a.layers:

        for value in np.nditer(layer.get_weights()):
            chromosome.append(value)

    return [[float(j) for j in i][0] for i in chromosome]
def chromosometomodel(chromosome):
    # newarray = np.array([(np.array([i],dtype = "float32"),np.array([0],dtype = "float32")) for i in chromosome])
    # newarray.reshape()
    newarray = np.array([np.array([chromosome[i] for i in range(20)]),np.array([chromosome[i] for i in range(20,30)])])
    newarray[0] = newarray[0].reshape(4,5)
    newarray[1] = newarray[1].reshape(5,2)
    return newarray
def newgen(topweights,randomweights,rate):
    newweights= []
    for i in range(len(topweights)):
        newweights.append((topweights[i],randomweights[i])[random.randint(0,1)])

    copy = []
    for i in newweights:
        t = i+ random.uniform(-0.5,0.5) * int(random.random() < rate)
        if t > 1 or t < 0:
            copy.append(i)
        else:
            copy.append(t)

    return copy


#[0.9674444997229463, 2.0, 30.0, 39.96872838099779, 50.0, 60.04208150203916, 69.99829680152149, 7.994611027244063, 9.0]

for _ in range(population):
    a = keras.models.Sequential()
    a.add(keras.Input(shape = (4,)))
    [a.add(layers.Dense(5,activation = "relu")) for i in range(1)]
    a.add(layers.Dense(2, activation="softmax"))

    a.compile(optimizer = "adam", loss = "categorical_crossentropy")
    networks.append(a)



lists = modeltochromosome(networks[0])
print(lists)
print(chromosometomodel(lists))
newweights = chromosometomodel(lists)

networks[0].layers[0].set_weights([newweights[0],np.zeros(5,)])
networks[0].layers[1].set_weights([newweights[1],np.zeros(2,)])

timer = time.time()
def train(networks):
    global steps
    global allscores

    allscores = {}
    for network in networks:
        score = 0
        env.reset()
        done = False
        action = [[0]]
        while not done:

            observation, reward, done, info = env.step(list(action[0]).index(max(action[0])))
            if score % 2 == 0:
                action = network.predict(x=np.array([observation]))

            score += reward
            steps += 1
            if abs(observation[2]) > math.pi / 3:
                observation = env.reset()
        allscores[network] = score
    return allscores



def scoregeneration(individuals,cores):

    splits = [[] for i in range(cores)]
    for i in range(len(individuals)):
        splits[i % cores].append(individuals[i])
    with Pool(cores) as p:
        p.map(train,splits)
    # processes = []
    # for i in range(cores):
    #     processes.append(Process(target = train,args = (splits[i],)))
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()


if __name__ == '__main__':
    steps = 0
    for g in range(generations):
        allscores = {}


        scoregeneration(networks,1)


        maxs = [0,0]
        print("Generation "+str(g+1) + "- Average time alive: " + str(sum(list(allscores.values()))/population))
        networ = []
        for network in allscores:
          if allscores[network] >= maxs[1]:
              networ.append(network)
              maxs = [network,allscores[network]]

        networks.remove(maxs[0])
        top = maxs[0]
        randoms=  networ[-1]
        topweights = modeltochromosome(top)
        randomweights = modeltochromosome(randoms)
        a = keras.models.Sequential()
        a.add(keras.Input(shape=(4,)))
        [a.add(layers.Dense(5, activation="relu")) for i in range(1)]
        a.add(layers.Dense(2, activation="softmax"))

        a.compile(optimizer="adam", loss="categorical_crossentropy")
        networks.append(a)

        for i in range(population):
            new = newgen(topweights,randomweights,0.2)
            convertednew = chromosometomodel(new)
            networks[i].layers[0].set_weights([convertednew[0],np.zeros(5,)])
            networks[i].layers[1].set_weights([convertednew[1],np.zeros(2,)])
        networks[0] = top


    print(steps/(-timer+time.time()))
    env.close()