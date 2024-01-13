# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
   
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        
        if len(self.keys()) == 0:
            return None
        total = list(self.items())
        values = [x[1] for x in total]
        bigI = values.index(max(values))
        return total[bigI][0]

    def total(self):
        
        return float(sum(self.values()))

    def normalize(self):

        sum = self.total()
        if sum == 0:
            return
        for key in self.keys():
            self[key] /= sum

    def sample(self):

        self.normalize()
        sp = random.random()
        for ind in self.keys():
            if(sp < self[ind]):
                return ind
            sp -= self[ind]

class InferenceModule:

    def __init__(self, ghostAgent):
 
        self.ghostAgent = ghostAgent
        self.placer = ghostAgent.index
        self.looker = []  

    def getJailPosition(self):
        return (2*self.ghostAgent.index - 1, 
                1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            uhoh = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            uhoh = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  
        discdist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            discdist[uhoh] = 1.0
            return discdist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  
        if ghostPosition in pacmanSuccessorStates: 
            track = 1.0 / float(len(pacmanSuccessorStates))
            discdist[uhoh] = track
        else:
            track = 0.0
        actionDist = agent.getDistribution(gameState)
        
        
        
        for doingstuff, probbob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, doingstuff)
            if successorPosition in pacmanSuccessorStates:  
                avariable = float(len(actionDist))
                discdist[uhoh] += probbob * (1.0/avariable) * (1.0 - track)
                discdist[successorPosition] = probbob * ((avariable - 1.0) / avariable) * (1.0 - track)
            else:
                discdist[successorPosition] = probbob * (1.0 - track)
        return discdist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):

        if index == None:
            index = self.placer - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        if(noisyDistance is None):
            if(jailPosition != ghostPosition):
                return 0
            return 1
        elif jailPosition == ghostPosition:
            return 0
        return busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))

    def setGhostPosition(self, gameState, ghostPosition, index):

        maked = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(maked, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):

        for index, pos in enumerate(ghostPositions):
            maked = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(maked, False)
        return gameState

    def observe(self, gameState):

        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.placer:  
            looker = distances[self.placer - 1]
            self.looker = looker
            
            
            
            self.observeUpdate(looker, gameState)

    def initialize(self, gameState):
        
        self.legalPositions = [weewoo for weewoo in gameState.getWalls().asList(False) if weewoo[1] > 1]
        
        
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        
        self.initializeUniformly(gameState)


    def initializeUniformly(self, gameState):

        raise NotImplementedError

    def observeUpdate(self, observation, gameState):

        raise NotImplementedError

    def elapseTime(self, gameState):

        raise NotImplementedError

    def getBeliefDistribution(self):

        raise NotImplementedError


class ExactInference(InferenceModule):

    def initializeUniformly(self, gameState):

        self.believer = DiscreteDistribution()
        for weewoo in self.legalPositions:
            self.believer[weewoo] = 1.0
            
            
        self.believer.normalize()
    def observeUpdate(self, observation, gameState):

        believer_copy = self.believer.copy()
        for key in self.believer.keys():
            self.believer[key] = believer_copy[key] * self.getObservationProb(observation, gameState.getPacmanPosition(), key, self.getJailPosition())
        self.believer.normalize()

    def elapseTime(self, gameState):

        all_dists = {}
        believer_copy = self.believer.copy()
        for notherenomore in self.allPositions:
            all_dists[notherenomore] = self.getPositionDistribution(gameState, notherenomore)
        
        for weewoo in self.allPositions:
            sum = 0
            for notherenomore in self.allPositions:
                sum += believer_copy[notherenomore] * all_dists[notherenomore][weewoo]
            self.believer[weewoo] = sum


    def getBeliefDistribution(self):
        
        
        return self.believer


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):

        self.pp = []
        legalmoves = len(self.legalPositions)
        allowed = self.legalPositions.copy()
        for r in range(self.numParticles):
            idx = int(random.random() * (len(allowed) - 1))
            self.pp.append(allowed[idx])
            allowed.pop(idx)
            if(len(allowed) == 0):
                allowed = self.legalPositions.copy()

    def observeUpdate(self, observation, gameState):

        discdist = DiscreteDistribution()
        for particle in self.pp:
            if(particle in discdist):
                discdist[particle] += self.getObservationProb(observation, gameState.getPacmanPosition(), particle, self.getJailPosition())
            else:
                discdist[particle] = self.getObservationProb(observation, gameState.getPacmanPosition(), particle, self.getJailPosition())
        if(discdist.total() == 0):
            self.initializeUniformly(gameState)
            discdist = self.getBeliefDistribution()
        self.pp = []
        discdist.normalize()
        for _ in range(self.numParticles):
            self.pp.append(discdist.sample())


    def elapseTime(self, gameState):
 
        particles_copy = self.pp.copy()
        self.pp = []
        for particle in particles_copy:
            discdist = self.getPositionDistribution(gameState, particle)
            self.pp.append(discdist.sample())

    def getBeliefDistribution(self):

        discdist = DiscreteDistribution()
        counts = {}
        for particle in self.pp:
            if(particle in counts):
                counts[particle] += 1
            else:
                counts[particle] = 1
        for position in counts.keys():
            discdist[position] = counts[position] / self.numParticles
        return discdist


class JointParticleFilter(ParticleFilter):

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.ghostshere = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):

        self.pp = []
        tuples = list(itertools.product(self.legalPositions, self.legalPositions))
        random.shuffle(tuples)
        for i in range(self.numParticles):
           self.pp.append(tuples[i % len(tuples)])

        

    def addGhostAgent(self, agent):

        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):

        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
 
        heft = DiscreteDistribution()
        for particle in self.pp:
            probbob = 1
            if(particle not in heft):
                heft[particle] = 0
            for i in range(self.ghostshere):
                probbob *= self.getObservationProb(observation[i], gameState.getPacmanPosition(), particle[i], self.getJailPosition(i))
            heft[particle] += probbob
        if(heft.total() == 0):
            self.initializeUniformly(gameState)
            heft = self.getBeliefDistribution()
        self.pp = []
        heft.normalize()
        for _ in range(self.numParticles):
            self.pp.append(heft.sample())


    def elapseTime(self, gameState):

        dalist = []
        for garbage in self.pp:
            newstuff = list(garbage)  # A list of ghost positions

            
            for i in range(self.ghostshere):
                discdist = self.getPositionDistribution(gameState, garbage, i, self.ghostAgents[i])
                newstuff[i] = discdist.sample()
            dalist.append(tuple(newstuff))
        self.pp = dalist



jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):

    def initializeUniformly(self, gameState):

        if self.placer == 1:
            
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.placer == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.placer == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):

        jointDistribution = jointInference.getBeliefDistribution()
        discdist = DiscreteDistribution()
        for dat, probbob in jointDistribution.items():
            discdist[dat[self.placer - 1]] += probbob
        return discdist