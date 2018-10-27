#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:49:45 2018

@author: tirth
"""

from Bipedal import Agent

agent = Agent()

#agent.train(1000)
#agent.save()
agent.load("weights.pkl")
agent.play(1)

