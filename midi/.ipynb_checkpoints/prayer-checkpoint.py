import numpy as np
import re

from .dictionary import FREQ, DUR

def music_str_parser(semantic):
    # finds string associated with symb
    found_str = re.compile(r'((note|gracenote|rest|multirest)(\-)(\S)*)'
                           ).findall(semantic)
    music_str = [i[0] for i in found_str]
    # finds the note's alphabets 
    fnd_notes = [re.compile(r'(([A-G](b|#)?[1-6])|rest)'
                    ).findall(note) for note in music_str]
    # stores the note's alphabets
    notes = [m[0][0] for m in fnd_notes]
    found_durs = [re.compile(r'((\_|\-)([a-z]|[0-9])+(\S)*)+'
                    ).findall(note) for note in music_str]
    #split by '_' every other string in list found in tuple of lists 
    durs = [i[0][0][1:].split('_') for i in found_durs]
    return notes, durs
def dur_evaluator(durations):
    note_dur_computed = []
    for dur in durations:
        # if dur_len in DUR dict, get. Else None 
        dur_len = [DUR.get(i.replace('.','').replace('.',''), 
                              None) for i in dur]
        # filter/remove None values, and sum list
        dur_len_actual = sum(list(filter(lambda a: a !=None, 
                                      dur_len)))
        # actual duration * 4 = quadruple
        if 'quadruple' in dur:
            dur_len_actual = dur_len_actual * 4
        # actual duration * 2 = fermata
        elif 'fermata' in dur:
            dur_len_actual = dur_len_actual * 2
        # actual duration + 1/2 of duration = .
        elif '.' in ''.join(dur):
            dur_len_actual = dur_len_actual + (dur_len_actual * 1/2)
        elif '..' in ''.join(dur):
            dur_len_actual = dur_len_actual +(2 *(dur_len_actual * 1/2))
        # if no special duration string
        elif dur[0].isnumeric():
            dur_len_actual = float(dur[0]) * .5
        note_dur_computed.append(dur_len_actual)
    return note_dur_computed
def get_music_note(semantic):
    notes, durations = music_str_parser(semantic)
    sample_rate = 44100
    timestep = []
    T = dur_evaluator(durations)
    for i in T:
        # gets timestep for each sample 
        timestep.append(np.linspace(0, i, int(i * sample_rate), 
                                    False))
    def get_freq(notes):
        # get pitchs frequency from dict
        pitch_freq = [FREQ[i] for i in notes]
        return pitch_freq
    return timestep, get_freq(notes)


def get_sinewave_audio(semantic):
    audio = []
    timestep, freq = get_music_note(semantic)
    for i in range(len(freq)):
        # calculates the sinewave
        audio.append(np.sin(
            freq[i] * timestep[i] * 2 * np.pi))
    return audio