#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:50:37 2020
"""
import sys,os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import py_midicsv
import numpy as np
import pandas as pd

def midi_to_csv(midi_filename:str) -> pd.DataFrame:
    csv_string = py_midicsv.midi_to_csv(midi_filename)

    data = []
    for row in csv_string:
        row = row.replace('\n','')
        row = row.split(', ')
        data += [row]

    return pd.DataFrame(data)


def csv_to_notes(data:pd.DataFrame) -> np.ndarray:
    cond = data[2] == 'Note_on_c'
    notes = data[cond][4]
    return notes.to_numpy()

def notes_to_midi(midi_filename:str, on_notes:np.ndarray, new_mini_name:str):
    csv_string = py_midicsv.midi_to_csv(midi_filename)
    N = len(csv_string)

    # make original ON_note and OFF_list list
    on_original  = []
    off_original = []

    for i in range(N):
        row = csv_string[i]
        row = row.replace('\n','')
        row = row.split(', ')

        if row[2] == 'Note_on_c':
            on_original += [row[4]]

        if row[2] == 'Note_off_c':
            off_original += [row[4]]

    # compute index map
    index_map = []
    for i in range(len(off_original)):
        first_one = True
        for j in range(len(on_original)):
            if off_original[i] == on_original[j] and first_one:
                first_one = False

                key_pressed = on_original[j]
                on_original[j] = -1

                index_map += [j]

    # conpute new off nodes
    off_notes = []
    for i in index_map:
        off_notes += [on_notes[i]]

    # replace notes
    data = []
    on_counts = 0
    off_counts= 0

    for i in range(len(csv_string)):
        row = csv_string[i]
        row = row.replace('\n','')
        row = row.split(', ')

        if row[2] == 'Note_on_c':
            row[4] = str(on_notes[on_counts])
            csv_string[i] = ', '.join(row) + '\n'
            on_counts += 1

        if row[2] == 'Note_off_c':
            row[4] = str(off_notes[off_counts])
            csv_string[i] = ', '.join(row) + '\n'
            off_counts += 1

    midi_object = py_midicsv.csv_to_midi(csv_string)

    with open(new_mini_name+'.mid', "wb") as output_file:
        midi_writer = py_midicsv.FileWriter(output_file)
        midi_writer.write(midi_object)


















