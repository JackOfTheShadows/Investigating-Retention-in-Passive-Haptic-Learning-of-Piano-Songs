from mido import MidiFile, tick2second
import sys
import glob
import os
from math import isinf
from numpy import array, zeros, full, argmin, inf, ndim
import numpy as np
from natsort import natsorted
import pandas as pd
from scipy  import stats
import math

retentionA = MidiFile('RetentionPhraseA.mid', clip=True)
retentionB = MidiFile('RetentionPhraseB-10.mid', clip=True)

noter_array_original = [72,74,76,77,79]

def printMid(filename):
    mid = MidiFile(''+filename, clip=True)
    print(mid)

    for msg in mid.tracks[0]:
        print(msg)
        
def midiToArrayVer1(mid):
#     counter = 1
    for i, track in enumerate(mid.tracks):
        notesArray = []
#         print('Track {}: {}'.format(i, track.name))
        for msg in track:
#            print(msg)
            if (msg.type == "note_on" and msg.time != 0):
#                 print(msg)
#                 if msg.note in noter_array_original:
                notesArray.append(msg.note)
#                 else:
#                     print(f"note is {msg.note} {mid}")
#                     print(f"counter {counter}")
#             counter = counter + 1
        return notesArray
    
def midiToArrayVer2(mid):
#     counter = 1
    for i, track in enumerate(mid.tracks):
        notesArray = []
#         print('Track {}: {}'.format(i, track.name))
        for msg in track:
#             print(msg)
            if (msg.type == "note_on" and msg.velocity > 0):
#                 print(msg)
#                 if msg.note in noter_array_original:
                notesArray.append(msg.note)
#                 else:
#                     print(f"note is {msg.note}")
#                     print(f"counter {counter}")
#             counter = counter + 1
        return notesArray
    
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    return D1[-1, -1]

custom_distance = lambda x, y: 0 if x==y else 1
def chord_distnce(x, y, len_chord):
    if x==y:
        return 0
    else:
        return (1 / len_chord)

def chooseSong(number):
    if (number % 2) == 0:
        activeSong = retentionB
        passiveSong = retentionA
    else:
        activeSong = retentionA
        passiveSong = retentionB
    return activeSong, passiveSong
    
def learn(number, condition):
    activeSong, passiveSong = chooseSong(number)
    if (condition == "active"):
        song = activeSong
    else:
        song = passiveSong
    attemptsArray = []
    i = 1
    last = 100
    holder = 100
    for name in natsorted(glob.glob(f'retentionData/{number}/{condition}Learn/*.mid')):
        filename = os.path.basename(name)
        (file, ext) = os.path.splitext(filename)
        if (file == "100"):
            continue
        mid = MidiFile(name, clip=True)
        test = dtw(midiToArrayVer1(mid), midiToArrayVer2(song), dist=custom_distance)
#         print(f"{filename} : {test}")
        if (test < last):
            last=test
            holder=last
        if(i == 3):
            attemptsArray.append(last)
            i = 1
            holder = last
            last = 100
        else:
            i = i+1
    return attemptsArray, holder

def first(number, condition):
    activeSong, passiveSong = chooseSong(number)
    if (condition == "active"):
        song = activeSong
    else:
        song = passiveSong
    for name in natsorted(glob.glob(f'retentionData/{number}/{condition}Learn/100.mid')):
        filename = os.path.basename(name)
        (file, ext) = os.path.splitext(filename)
        if (file == "100"):
            mid = MidiFile(name, clip=True)
            return dtw(midiToArrayVer1(mid), midiToArrayVer2(song), dist=custom_distance)
    return 5

def aidedTest(number, condition, aided=None):
    ###setup active passive aided not aided
    activeSong, passiveSong = chooseSong(number)
    if (condition == "active"):
        song = activeSong
    else:
        song = passiveSong
    if aided:
        aid = "-aided"
    else:
        aid = ""
    value = None
    for name in natsorted(glob.glob(f'retentionData/{number}/{condition}/session2{aid}/*.mid')):
        filename = os.path.basename(name)
        (file, ext) = os.path.splitext(filename)
        mid = MidiFile(name, clip=True)
        test = dtw(midiToArrayVer1(mid), midiToArrayVer2(song), dist=custom_distance)
#         print(f"{filename} : {test}")
        if(value is None):
            value = test
        if (test < value):
            value=test 
    if value is None:
        value = 0
    return value
        
def activeLearn(number):
    return learn(number, "active")
            
def passiveLearn(number):
    return learn(number, "passive")
        
def passiveFirst(number):
    return first(number, "passive")

def activeFirst(number):
    return first(number, "active")

def activeFirstTest(number):
    return aidedTest(number, "active")
            
def passiveFirstTest(number):
    return aidedTest(number, "passive")
    
def activeAidedTest(number):
    return aidedTest(number, "active", "aided")
            
def passiveAidedTest(number):
    return aidedTest(number, "passive", "aided")

#creating a pandas data frame
activeLearnTime_all = {
  2: 2,
  4: 1,
  5: 3,
  6: 2,
  7: 2,
  8: 3,
  9: 2,
  10: 7,
  11: 4,
  12: 2,
  13: 11,
  14: 2,
  15: 1,
  16: 3,
  17: 6,
  18: 2,
  19: 8,
  20: 6,
  21: 4,
  22: 4,
  23: 13,
  24: 1
}

experience_all = {
    2: 5,
    4: 0,
    5: 0,
    6: 0.5,
    7: 0.5,
    8: 2,
    9: 5,
    10: 0, 
    11: 10,
    12: 0, 
    13: 0, 
    14: 10,
    15: 0.5,
    16: 3, 
    17: 1, 
    18: 0.1,
    19: 0, 
    20: 2, 
    21: 0, 
    22: 0, 
    23: 0, 
    24: 0.1}

def generateDataFrame(participents):
    active = []
    active_aided = []
    passive = []
    passive_aided = []
    active_learn = []
    passive_learn = []
    active_first = []
    passive_first = []
    active_time = []
    passive_time = []
    experience = []
    #new
    active_attempts = []
    passive_attempts = []
    for n in participents:
        active.append(activeFirstTest(n))
        passive.append(passiveFirstTest(n))
        active_aided.append(activeAidedTest(n))
        passive_aided.append(passiveAidedTest(n))
        active_attempts_learn_holder, holder = activeLearn(n)
        active_learn.append(holder)
        passive_learn_holder, holder = passiveLearn(n)
        passive_learn.append(holder)
        holder = activeFirst(n)
        active_first.append(holder)
        holder = passiveFirst(n)
        passive_first.append(holder)
        active_time.append(activeLearnTime_all[n])
        passive_time.append(len(passive_learn_holder)*20)
        experience.append(experience_all[n])
        active_attempts.append(len(active_attempts_learn_holder))
        passive_attempts.append(len(passive_learn_holder))
        
    errorsActiveFT = []
    errorsPassiveFT = []
    errorsActiveAT = []
    errorsPassiveAT = []
    for n in participents:
        _, activeL = activeLearn(n)
        activeFT = activeFirstTest(n)
        activeAT = activeAidedTest(n)
        _, passiveL = passiveLearn(n)
        passiveFT = passiveFirstTest(n)
        passiveAT = passiveAidedTest(n)
        peFT = passiveFT - passiveL
        aeFT = activeFT - activeL
        errorsActiveFT.append(aeFT)
        errorsPassiveFT.append(peFT)

        peAT = passiveAT - passiveL
        aeAT = activeAT - activeL
        errorsActiveAT.append(aeAT)
        errorsPassiveAT.append(peAT)
        
    data = {'active' : active,
            'active aided' : active_aided,
            'passive' : passive,
            'passive aided' : passive_aided,
            'active learn' : active_learn,
            'passive learn' : passive_learn,
            'active first': active_first,
            'passive first' : passive_first,
            'active time' : active_time,
            'passive time' : passive_time,
            'experience': experience,
            'errorsActiveFT': errorsActiveFT,
            'errorsActiveAT': errorsActiveAT,
            'errorsPassiveFT': errorsPassiveFT,
            'errorsPassiveAT': errorsPassiveAT,
            'activeAttempts' : active_attempts,
            'passiveAttempts' : passive_attempts}
    df = pd.DataFrame(data, columns=['active', 
                                     'active aided', 
                                     'passive', 
                                     'passive aided',
                                     'active learn',
                                     'passive learn',
                                     'active first',
                                     'passive first',
                                     'active time',
                                     'passive time',
                                     'experience',
                                     'errorsActiveFT',
                                     'errorsActiveAT',
                                     'errorsPassiveFT',
                                     'errorsPassiveAT',
                                     'activeAttempts',
                                     'passiveAttempts'])
    return df

def gen_stats(df_frame, groupA, groupB):
    print("--------MEAN ALL------------")
    print(f"mean of {groupA}: {np.around(np.mean(df_frame[groupA]), decimals=3)}")
    print(f"mean of {groupB} : {np.around(np.mean(df_frame[groupB]), decimals=3)}")
    print("------------------------")

    print("--------std ALL------------")
    print(f"standard diviation of {groupA}: {np.around(np.std(df_frame[groupA]), decimals=3)}")
    print(f"standard diviation of {groupB}: {np.around(np.std(df_frame[groupB]), decimals=3)}")
    print("------------------------")

    print("--------SE ALL------------")
    print(f"standard error of {groupA} : {np.around(stats.sem(df_frame[groupA], axis=None, ddof=0), decimals=3)}")
    print(f"standard error of {groupB}: {np.around(stats.sem(df_frame[groupB], axis=None, ddof=0), decimals=3)}")
    print("------------------------")

    print("--------Normal distribution ALL------------")
    print(f"Gaussian: {groupA}: {np.around(stats.shapiro(df_frame[groupA]), decimals=3)}")
    print(f"Gaussian: {groupB}: {np.around(stats.shapiro(df_frame[groupB]), decimals=3)}")
    print("------------------------")


def primacy(number, attempt):
    activeSong, passiveSong = chooseSong(number)
    activeName = f'retentionData/{number}/activeLearn/{attempt}.mid'
    passiveName = f'retentionData/{number}/passiveLearn/{attempt}.mid'
    activeMid = MidiFile(activeName, clip=True)
    passiveMid = MidiFile(passiveName, clip=True)
    activePlayedArray = midiToArrayVer1(activeMid)
    activeSongArray = midiToArrayVer2(activeSong)
    passivePlayedArray = midiToArrayVer1(passiveMid)
    passiveSongArray = midiToArrayVer2(passiveSong)
    i = 0
    equal = True
    while equal:
        if activePlayedArray[i] == activeSongArray[i]:
            i = i + 1
            if i == len(activePlayedArray) or i == len(activeSongArray):
                equal = False
        else:
            equal = False
#     print(f"user {number} active primacy = {i}")
    activePrimacy = i
    i = 0
    equal = True
    while equal:
        if passivePlayedArray[i] == passiveSongArray[i]:
            i = i + 1
            if i == len(passivePlayedArray) or i == len(passiveSongArray):
                equal = False
        else:
            equal = False
#     print(f"user {number} passive primacy = {i}")   
    passivePrimcacy = i
    return activePrimacy, passivePrimcacy

def recency(number, attempt):
    activeSong, passiveSong = chooseSong(number)
    activeName = f'retentionData/{number}/activeLearn/{attempt}.mid'
    passiveName = f'retentionData/{number}/passiveLearn/{attempt}.mid'
    activeMid = MidiFile(activeName, clip=True)
    passiveMid = MidiFile(passiveName, clip=True)
    activePlayedArray = midiToArrayVer1(activeMid)
    activeSongArray = midiToArrayVer2(activeSong)
    passivePlayedArray = midiToArrayVer1(passiveMid)
    passiveSongArray = midiToArrayVer2(passiveSong)
#     print(activePlayedArray)
#     print(activeSongArray)
#     print(passivePlayedArray)
#     print(passiveSongArray)
    i = 0
    equal = True
    a = len(activePlayedArray) -1
    b = len(activeSongArray) -1
    while equal:
        if activePlayedArray[a] == activeSongArray[b]:
            i = i + 1
            a = a - 1
            b = b -1
            if a < 0 or b < 0:
                equal = False
        else:
            equal = False
#     print(f"user {number} active primacy = {i}")
    activePrimacy = i
    i = 0
    equal = True
    a = len(passivePlayedArray) -1
    b = len(passiveSongArray) -1
    while equal:
        if passivePlayedArray[a] == passiveSongArray[b]:
            i = i + 1
            if i == len(passivePlayedArray) or i == len(passiveSongArray):
                equal = False
        else:
            equal = False
#     print(f"user {number} passive primacy = {i}")   
    passivePrimcacy = i
    return activePrimacy, passivePrimcacy

#creating a pandas data frame

def generateDataFrame_memory(participents):
    activePrimacy = []
    passivePrimacy = []
    activeRecency = []
    passiveRecency = []
    activePrimacy2 = []
    passivePrimacy2 = []
    activeRecency2 = []
    passiveRecency2 = []
    activePrimacy3 = []
    passivePrimacy3 = []
    activeRecency3 = []
    passiveRecency3 = []
    for n in participents:
        a, p = primacy(n,1)
        activePrimacy.append(a)
        passivePrimacy.append(p)
        ar, pr = recency(n,1)
        activeRecency.append(ar)
        passiveRecency.append(pr)
        
    for n in participents:
        a, p = primacy(n,2)
        activePrimacy2.append(a)
        passivePrimacy2.append(p)
        ar, pr = recency(n,2)
        activeRecency2.append(ar)
        passiveRecency2.append(pr)
        
    for n in participents:
        a, p = primacy(n,3)
        activePrimacy3.append(a)
        passivePrimacy3.append(p)
        ar, pr = recency(n,3)
        activeRecency3.append(ar)
        passiveRecency3.append(pr)
        
    data = {'activePrimacy' : activePrimacy,
            'passivePrimacy' : passivePrimacy,
           'activeRecency' : activeRecency,
            'passiveRecency' : passiveRecency,
           'activePrimacy2' : activePrimacy2,
            'passivePrimacy2' : passivePrimacy2,
           'activeRecency2' : activeRecency2,
            'passiveRecency2' : passiveRecency2,
           'activePrimacy3' : activePrimacy3,
            'passivePrimacy3' : passivePrimacy3,
           'activeRecency3' : activeRecency3,
            'passiveRecency3' : passiveRecency3}
    df = pd.DataFrame(data, columns=['activePrimacy', 
                                     'passivePrimacy',
                                     'activeRecency',
                                     'passiveRecency',
                                     'activePrimacy2', 
                                     'passivePrimacy2',
                                     'activeRecency2',
                                     'passiveRecency2',
                                     'activePrimacy3', 
                                     'passivePrimacy3',
                                     'activeRecency3',
                                     'passiveRecency3'])
    return df
