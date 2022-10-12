import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/evan/Projects/sequencer')

from sequencer import *

if __name__ == "__main__":
    pent_wf = csv_bpm("pent.csv")
    pent_wf.write("pent.wav")
    write_csv("pent2")
    write_csv("pent3")
    write_csv("pent_drone")

#    verse_a = csv_bpm("drums2.csv")
#    verse_a.write("verse.wav")
#    verse_b = csv_bpm("drums_verse_2.csv")
#    verse_seq = Sequencer(16, seq_bpm)
#    verse_seq.place_waveform(0, verse_a)
#    verse_seq.place_waveform(8, verse_b)
#    verse_fill = csv_bpm("verse_fill.csv")
#    verse_fill_seq = Sequencer(16,seq_bpm)
#    verse_fill_seq.place_waveform(0, verse_a)
#    verse_fill_seq.place_waveform(8, verse_fill)
#    full_verse = Sequencer(65,seq_bpm)
#    full_verse.place_waveform(0, verse_seq)
#    full_verse.place_waveform(16, verse_seq)
#    full_verse.place_waveform(32, verse_seq)
#    full_verse.place_waveform(48, verse_fill_seq)
#    full_verse.write("verse.wav")
#
#    chorus_seq_a = Sequencer(16,seq_bpm)
#    chorus_b = csv_bpm("chorus_b.csv")
#    chorus_b.write("chorus_b.wav")
#    chorus_seq_a.place_waveform(0, verse_a)
#    chorus_seq_a.place_waveform(8, chorus_b)
#    chorus_seq_a.write("chorus_b.wav")
#
#    chorus_seq_b = Sequencer(32,seq_bpm)
#    chorus_fill = csv_bpm("chorus_fill.csv")
#    chorus_seq_b.place_waveform(0, verse_a)
#    chorus_seq_b.place_waveform(8, chorus_fill)
#
#    full_chorus = Sequencer(80, seq_bpm)
#    full_chorus.place_waveform(0, chorus_seq_a)
#    full_chorus.place_waveform(16, chorus_seq_a)
#    full_chorus.place_waveform(32, chorus_seq_a)
#    full_chorus.place_waveform(48, chorus_seq_b)
#    full_chorus.write("full_chorus.wav")
#
#    loaded = load_csv("drums_verse_2.csv")

#    #loaded.play()
#    loaded.plot()
#    loaded.write("drums.wav")
