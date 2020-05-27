#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:51:54 2020
"""
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import yaml

#EV Config class
class EV_Config:
    """
    EV configuration class
    """
    # class variables
    sectionName='EV'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'nHiddenStates':(int,True),
             'nObservableStates':(int,True),
             'learningRate':(float,True),
			 'useFullSequence':(bool,True)
			 }

    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.safe_load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))

        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]

                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))

                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)

    #string representation for class data
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))

