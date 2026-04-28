"""
===============================================================================
Project:        Data Science Final Project: Online Poker Behavioral Analysis
Authors:        Joshua Myers, German Garrido-Lestache Belinchon
Description:    Contains the core Monte Carlo simulation engine. Filters for hands 
                where a player folded, and runs PokerKit simulations (1,000+ runouts) 
                to determine how often they would have won. Calculates the 
                Expected Value (EV) sacrificed by folding.
Dependencies:   pokerkit, random, numpy
Integration:    Imported in `main.py`
===============================================================================
"""