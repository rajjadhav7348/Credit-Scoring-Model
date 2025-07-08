
import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model

def generate_music():
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)

    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    n_vocab = len(set(notes))
    network_input = []
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])

    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    model = load_model("weights/your-best-weight.hdf5")  # replace with actual filename

    prediction_output = []
    for note_index in range(500):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    output_notes = []
    offset = 0
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in chord_notes: n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')

if __name__ == '__main__':
    generate_music()
