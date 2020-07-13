#!/usr/bin/env python

# Postprocessing performs basic algorithms on Redis data
#
# This software is part of the EEGsynth project, see <https://github.com/eegsynth/eegsynth>.
#
# Copyright (C) 2017-2020 EEGsynth project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from numpy import log, log2, log10, exp, power, sqrt, mean, median, var, std, mod, random
import configparser
import argparse
import numpy as np
import os
import redis
import sys
import time

if hasattr(sys, 'frozen'):
    path = os.path.split(sys.executable)[0]
    file = os.path.split(sys.executable)[-1]
    name = os.path.splitext(file)[0]
elif __name__=='__main__' and sys.argv[0] != '':
    path = os.path.split(sys.argv[0])[0]
    file = os.path.split(sys.argv[0])[-1]
    name = os.path.splitext(file)[0]
elif __name__=='__main__':
    path = os.path.abspath('')
    file = os.path.split(path)[-1] + '.py'
    name = os.path.splitext(file)[0]
else:
    path = os.path.split(__file__)[0]
    file = os.path.split(__file__)[-1]
    name = os.path.splitext(file)[0]

# eegsynth/lib contains shared modules
sys.path.insert(0, os.path.join(path,'../../lib'))
import EEGsynth

# these function names can be used in the equation that gets parsed
from EEGsynth import compress, limit, rescale, normalizerange, normalizestandard

### Biotuning toolbox
import numpy as np
import math
from fractions import Fraction
import itertools
max_sub = 120
octave_limit = 0.05
cons_thresh = 0.01

def rebound(x, low = 1, high = 2, octave = 2):
    while x > high:
        x = x/octave
    while x < low:
        x = x*octave
    return x

def nth_root (num, root):
    answer = num**(1/root)
    return answer


#Function that compares lists (i.e. peak harmonics)
def compareLists(list1, list2, bounds):
    matching = []
    matching_pos = []
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l2-bounds < l1 < l2+bounds:
                matching.append((l1+l2)/2)
                matching_pos.append([(l1+l2)/2, i+1, j+1])
    matching = np.array(matching)
    matching_pos = np.array(matching_pos)
    ratios_temp = []
    for i in range(len(matching_pos)):
        if matching_pos[i][1]>matching_pos[i][2]:
            ratios_temp.append(matching_pos[i][1]/matching_pos[i][2])
            #print(matching_pos[i][0])
        else:
            ratios_temp.append(matching_pos[i][2]/matching_pos[i][1])
    matching_pos_ratios = np.array(ratios_temp)
    return matching, matching_pos, matching_pos_ratios

#This function calculates the 15 ratios (with the possibility to bound them between 1 and 2) derived from the 6 peaks
def compute_peak_ratios(peaks, rebound = 1, octave = 2):
    ratios = []
    peak_ratios_rebound = []
    for p1 in peaks:
        for p2 in peaks:
            ratio_temp = p2/p1
            if ratio_temp == 1:
                ratio_temp = None
            elif ratio_temp < 1:
                ratio_temp = None

            ratios.append(ratio_temp)

    peak_ratios = np.array(ratios)
    peak_ratios = [i for i in peak_ratios if i]
    peak_ratios = list(set(peak_ratios))
    if rebound == 1:
        for peak in peak_ratios:
            while peak > octave:
                peak = peak/octave
            peak_ratios_rebound.append(peak)
    peak_ratios_rebound = np.array(peak_ratios_rebound)
    peak_ratios_rebound = [i for i in peak_ratios_rebound if i]
    peak_ratios = sorted(peak_ratios)
    peak_ratios_rebound = sorted(list(set(peak_ratios_rebound)))
    return peak_ratios, peak_ratios_rebound


#This function takes a list of frequency peaks as input and computes the desired number of harmonics
#with the formula: x + 2x + 3x ... + nx
def EEG_harmonics_mult(peaks, n_harmonics, n_oct_up = 0):
    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    multi_harmonics_rebound = []
    for p in peaks:
        multi_harmonics_r = []
        multi_harm_temp = []
        harmonics = []
        p = p * (2**n_oct_up)
        i = 1
        harm_temp = p
        while i < n_harmonics:
            harm_temp = p * i
            harmonics.append(harm_temp)
            i+=1
        multi_harmonics.append(harmonics)
    multi_harmonics = np.array(multi_harmonics)

    return multi_harmonics

#This function takes a list of frequency peaks as input and computes the desired number of harmonics
#with the formula: x + x/2 + x/3 ... + x/n
def EEG_harmonics_div(peaks, n_harmonics, n_oct_up = 0):
    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    multi_harmonics_sub = []
    for p in peaks:

        harmonics = []
        harmonics_sub = []
        p = p * (2**n_oct_up)
        i = 2
        harm_temp = p
        harm_temp_sub = p
        while i < n_harmonics:
            harm_temp = harm_temp + (p/i)
            harm_temp_sub = abs(harm_temp_sub - (p/i))
            harmonics.append(harm_temp)
            harmonics_sub.append(harm_temp_sub)
            i+=1
        multi_harmonics.append(harmonics)
        multi_harmonics_sub.append(harmonics_sub)
    multi_harmonics = np.array(multi_harmonics)
    multi_harmonics_bounded = multi_harmonics.copy()
    multi_harmonics_sub = np.array(multi_harmonics_sub)
    multi_harmonics_sub_bounded = multi_harmonics_sub.copy()
    #Rebound the result between 1 and 2
    for i in range(len(multi_harmonics_bounded)):
        for j in range(len(multi_harmonics_bounded[0])):
            multi_harmonics_bounded[i][j] = rebound(multi_harmonics_bounded[i][j])
            multi_harmonics_sub_bounded[i][j] = rebound(multi_harmonics_sub_bounded[i][j])
    return multi_harmonics, multi_harmonics_bounded, multi_harmonics_sub, multi_harmonics_sub_bounded

#This function computes harmonics of a list of peaks
def harmonic_fit(peaks, n_harm = 10, bounds = 1, function = 'mult'):
    from itertools import combinations
    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == 'mult':
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == 'div':
        multi_harmonics, x, y, z = EEG_harmonics_div(peaks, n_harm)
    #print(multi_harmonics)
    list_peaks = list(combinations(peak_bands,2))
    #print(list_peaks)
    harm_temp = []
    for i in range(len(list_peaks)):
        harms, b, c = compareLists(multi_harmonics[list_peaks[i][0]], multi_harmonics[list_peaks[i][1]], bounds)
        harm_temp.append(harms)
    harm_fit = np.array(harm_temp).squeeze()

    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))

    return harm_fit

#Oct_subdiv
#Argument 1 : a ratio in the form of a float or a fraction
#Argument 2 : bounds between which the octave should fall
#Argument 3 : value of the octave
#Argument 4 : number of octave subdivisions

def oct_subdiv(ratio,bounds,octave,n):
    Octdiv, Octvalue, i = [], [], 1
    ratios = []
    while len(Octdiv) < n:
        ratio_mult = (ratio**i)
        while ratio_mult > octave:
            ratio_mult = ratio_mult/octave

        rescale_ratio = ratio_mult - round(ratio_mult)
        ratios.append(ratio_mult)
        i+=1
        if -bounds < rescale_ratio < bounds:
            Octdiv.append(i-1)
            Octvalue.append(ratio_mult)
        else:
            continue
    return Octdiv, Octvalue, ratios



def compare_oct_div(Octdiv = 53, Octdiv2 = 12, bounds = 0.01, octave = 2):
    ListOctdiv = []
    ListOctdiv2 = []
    OctdivSum = 1
    OctdivSum2 = 1
    i = 1
    i2 = 1
    Matching_harmonics = []
    #HARMONIC_RATIOS = [1, 1.0595, 1.1225, 1.1892, 1.2599, 1.3348, 1.4142, 1.4983, 1.5874, 1.6818, 1.7818, 1.8897]
    while OctdivSum < octave:
        OctdivSum =(nth_root(octave, Octdiv))**i
        i+=1
        ListOctdiv.append(OctdivSum)
    #print(ListOctdiv)

    while OctdivSum2 < octave:
        OctdivSum2 =(nth_root(octave, Octdiv2))**i2
        i2+=1
        ListOctdiv2.append(OctdivSum2)
    #print(ListOctdiv2)
    for i, n in enumerate(ListOctdiv):
        for j, harm in enumerate(ListOctdiv2):
            if harm-bounds < n < harm+bounds:
                Matching_harmonics.append([n, i+1, harm, j+1])
    Matching_harmonics = np.array(Matching_harmonics)
    return Matching_harmonics


def compute_consonance (peaks, limit):
    consonance_ = []
    peaks2keep = []
    peaks_consonance = []
    for p1 in peaks:
        for p2 in peaks:
            peaks2keep_temp = []
            p2x = p2
            p1x = p1
            cons_temp = (p2x/p1x).as_integer_ratio()
            cons_temp = (cons_temp[0] + cons_temp[1])/(cons_temp[0] * cons_temp[1])
            #print(cons_temp)
            if cons_temp == 2:
                cons_temp = None
                p2x = None
                p1x = None
            elif cons_temp < limit:
                cons_temp = None
                p2x = None
                p1x = None
            if p2x != None:
                peaks2keep_temp.extend([p2x, p1x])
            consonance_.append(cons_temp)
            peaks2keep.append(peaks2keep_temp)
        peaks_consonance = np.array(peaks2keep)
        peaks_consonance = [x for x in peaks_consonance if x]
        consonance = np.array(consonance_)
        consonance = [i for i in consonance if i]

        #consonance = list(set(consonance))
    return consonance, peaks_consonance

# Function that computes integer ratios from peaks with higher consonance
# Needs at least two pairs of values
def comp_consonant_integers_ratios (peaks_consonance):
    cons_integer = []
    for i in range(len(peaks_consonance)):
        #for j in range(len(peaks_consonance[0])):
        a = peaks_consonance[i][0]
        b = peaks_consonance[i][1]
        if a > b:
            while a > (b):
                a = a/2
        if a < b:
            while b > (a):
                b = b/2
        c = a/b
        cons_temp = (c).as_integer_ratio()
        #print(cons_temp)
        cons_integer.append(cons_temp)
    consonant_integers = np.array(cons_integer).squeeze()
    cons_ratios = []
    for j in range(len(consonant_integers)):
        cons_ratios.append((consonant_integers[j][0])/(consonant_integers[j][1]))
    consonant_ratios = np.array(cons_ratios)
    cons_rat = []
    for i in consonant_ratios:
        if i not in cons_rat:
            cons_rat.append(i)
    try:
        cons_rat.remove(1.0)
    except (ValueError, TypeError, NameError):
        pass
        #cons_int = []
        #for i in consonant_integers:
        #    if i not in cons_int:
        #        cons_int.append(i)
    return consonant_integers, cons_rat

##Here we use the most consonant peaks ratios as input of oct_subdiv function. Each consonant ratio
##leads to a list of possible octave subdivisions. These lists are compared and optimal octave subdivisions are
##determined.

#Output1: octave subdivisions
#Output2: ratios that led to Output1

def multi_oct_subdiv (peaks, max_sub, octave_limit, cons_thresh):
    import itertools
    from collections import Counter
    a, b = compute_consonance(peaks, cons_thresh)
    c, ratios = comp_consonant_integers_ratios(b)

    list_oct_div = []
    for i in range(len(ratios)):
        list_temp, no, no2 = oct_subdiv(ratios[i], octave_limit, 2, 30)
        list_oct_div.append(list_temp)


    counts = Counter(list(itertools.chain(*list_oct_div)))
    oct_div_temp = []
    for k, v in counts.items():
        if v > 1:
            oct_div_temp.append(k)
    oct_div_temp = np.sort(oct_div_temp)
    oct_div_final = []
    for i in range(len(oct_div_temp)):
        if oct_div_temp[i] < max_sub:
            oct_div_final.append(oct_div_temp[i])
    return oct_div_final, ratios


def _setup():
    '''Initialize the module
    This adds a set of global variables
    '''
    global parser, args, config, r, response, patch

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inifile", default=os.path.join(path, name + '.ini'), help="name of the configuration file")
    args = parser.parse_args()

    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(args.inifile)

    try:
        r = redis.StrictRedis(host=config.get('redis', 'hostname'), port=config.getint('redis', 'port'), db=0, charset='utf-8', decode_responses=True)
        response = r.client_list()
    except redis.ConnectionError:
        raise RuntimeError("cannot connect to Redis server")

    # combine the patching from the configuration file and Redis
    patch = EEGsynth.patch(config, r)

    # there should not be any local variables in this function, they should all be global
    if len(locals()):
        print('LOCALS: ' + ', '.join(locals().keys()))


def _start():
    '''Start the module
    This uses the global variables from setup and adds a set of global variables
    '''
    global parser, args, config, r, response, patch, name
    global monitor, input_name, input_variable, variable

    # this can be used to show parameters that have changed
    monitor = EEGsynth.monitor(name=name, debug=patch.getint('general', 'debug'))

    # get the input and output options
    if len(config.items('input')):
        input_name, input_variable = list(zip(*config.items('input')))
    else:
        input_name, input_variable = ([], [])

    monitor.info('===== input variables =====')
    for name,variable in zip(input_name, input_variable):
        monitor.info(name + ' = ' + variable)
    monitor.info('============================')

    # there should not be any local variables in this function, they should all be global
    if len(locals()):
        print('LOCALS: ' + ', '.join(locals().keys()))


def _loop_once():
    '''Run the main loop once
    This uses the global variables from setup and start, and adds a set of global variables
    '''
    global parser, args, config, r, response, patch
    global monitor, input_name, input_variable

    monitor.debug('============================')

    input_value = []
    for name in input_name:
        val = patch.getfloat('input', name)
        print(val)
        input_value.append(val)

    _, harm_ratios = compute_peak_ratios(harmonic_fit(input_value, 50, 0.01, 'mult'))
    monitor.debug('Peak values : %s ================ Harmonic ratios %s' % (input_value, harm_ratios))
    patch.setvalue('harmratios', harm_ratios)

def _loop_forever():
    '''Run the main loop forever
    '''
    global monitor, patch
    while True:
        monitor.loop()
        _loop_once()
        time.sleep(patch.getfloat('general', 'delay'))


def _stop():
    '''Stop and clean up on SystemExit, KeyboardInterrupt
    '''
    sys.exit()


if __name__ == '__main__':
    _setup()
    _start()
    _loop_forever()
