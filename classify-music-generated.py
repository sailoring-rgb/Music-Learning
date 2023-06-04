import numpy as np
import os, IPython, music21
from music21 import note, chord, converter, instrument
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

class ClassifyMusic:

    def __init__(self, artist, filename):
         self.artist = artist
         self.filename = filename
         self.length = 40
        
    ### 1. Importação dos dados
    def importData(self):
        cwd = os.getcwd()
        if self.artist == "Beethoven":
            filepath = cwd + "/input/classical-music/beeth/"
        elif self.artist == "Halsey":
            filepath = cwd + "/input/halsey-music/"
        else:
            filepath = cwd + "/input/madonna-music/"

        self.midi_files = []
        for i in os.listdir(filepath):
            if i.endswith(".mid"):
                tr = filepath+i
                midi = converter.parse(tr)
                self.midi_files.append(midi)

    ### 1.2. Extração das características das notas
    def extract_notes(self, file):
        pitches = []
        durations = []
        steps = []
        octaves = []
        pick = None

        for j in file:
            songs = instrument.partitionByInstrument(j)
            for part in songs.parts:
                pick = part.recurse()
                for element in pick:
                    if isinstance(element, note.Note):
                        pitches.append(str(element.pitch))
                        durations.append(element.duration.quarterLength)
                        steps.append(str(element.pitch.step))
                        octaves.append(str(element.pitch.octave))
                    elif isinstance(element, chord.Chord):
                        pitches.append(".".join(str(n) for n in element.normalOrder))
                        durations.append(element.duration.quarterLength)
                        chord_steps = str(min(element.pitches).step)
                        chord_octave = str(min(element.pitches).octave)
                        steps.append(chord_steps)
                        octaves.append(chord_octave)

        return pitches, durations, steps, octaves

    ### 2. Preparação dos dados
    def prepareData(self):
        self.pitches, self.durations, self.steps, self.octaves = self.extract_notes(self.midi_files)

        symb_notes = sorted(list(set(self.pitches))) 
        symb_durations = sorted(list(set(self.durations))) 
        symb_steps = sorted(list(set(self.steps))) 
        symb_octaves = sorted(list(set(self.octaves))) 

        L_symb_notes = len(symb_notes) 
        self.L_symb_notes = len(symb_notes) 
        L_symb_durations = len(symb_durations) 
        L_symb_steps = len(symb_steps)
        L_symb_octaves = len(symb_octaves)

        mapping_notes = dict((c, i) for i, c in enumerate(symb_notes))  
        mapping_durations = dict((c, i) for i, c in enumerate(symb_durations))  
        mapping_steps = dict((c, i) for i, c in enumerate(symb_steps)) 
        mapping_octaves = dict((c, i) for i, c in enumerate(symb_octaves)) 
        L_pitches = len(self.pitches) 

        note_features = []
        duration_features = []
        step_features = []
        octave_features = []

        note_targets = []
        duration_targets = []
        step_targets = []
        octave_targets = []

        for i in range(0, L_pitches - self.length, 1):
            note_feature = self.pitches[i:i + self.length]
            duration_feature = self.durations[i:i + self.length]
            step_feature = self.steps[i:i + self.length]
            octave_feature = self.octaves[i:i + self.length]

            target_note = self.pitches[i + self.length]
            target_duration = self.durations[i + self.length]
            target_step = self.steps[i + self.length]
            target_octave = self.octaves[i + self.length]

            note_features.append([mapping_notes[j] for j in note_feature])
            duration_features.append([mapping_durations[k] for k in duration_feature])
            step_features.append([mapping_steps[q] for q in step_feature])
            octave_features.append([mapping_octaves[p] for p in octave_feature])

            note_targets.append(target_note)
            duration_targets.append(target_duration)
            step_targets.append(target_step)
            octave_targets.append(target_octave)

        L_datapoints = len(note_targets)
        note_features = (np.reshape(note_features, (L_datapoints, self.length, 1))) / float(L_symb_notes)
        duration_features = (np.reshape(duration_features, (L_datapoints, self.length, 1))) / float(L_symb_durations)
        step_features = (np.reshape(step_features, (L_datapoints, self.length, 1))) / float(L_symb_steps)
        octave_features = (np.reshape(octave_features, (L_datapoints, self.length, 1))) / float(L_symb_octaves)

        class_mapping = {"good": 0, "average": 1, "bad": 2}
        note_targets = np.array([class_mapping.get(note, -1) for note in note_targets])
        duration_targets = np.array([class_mapping.get(duration, -1) for duration in duration_targets])
        step_targets = np.array([class_mapping.get(step, -1) for step in step_targets])
        octave_targets = np.array([class_mapping.get(octave, -1) for octave in octave_targets])

        note_targets = note_targets[note_targets != -1]
        duration_targets = duration_targets[duration_targets != -1]
        step_targets = step_targets[step_targets != -1]
        octave_targets = octave_targets[octave_targets != -1]

        if len(note_targets) == 0 or len(duration_targets) == 0 or len(step_targets) == 0 or len(octave_targets) == 0:
            print("Não há amostras suficientes para treinamento.")
            return

        note_targets = tf.keras.utils.to_categorical(note_targets)
        duration_targets = tf.keras.utils.to_categorical(duration_targets)
        step_targets = tf.keras.utils.to_categorical(step_targets)
        octave_targets = tf.keras.utils.to_categorical(octave_targets)

        X_note_train, X_note_test, X_duration_train, X_duration_test, X_step_train, X_step_test, X_octave_train, X_octave_test, y_target_note_train, y_target_note_test, y_target_duration_train, y_target_duration_test, y_target_step_train, y_target_step_test, y_target_octave_train, y_target_octave_test = train_test_split(
            note_features, duration_features, step_features, octave_features, note_targets, duration_targets, step_targets, octave_targets, test_size=0.2, random_state=42
        )

        return L_symb_notes, X_note_train, X_note_test, X_duration_train, X_duration_test, X_step_train, X_step_test, X_octave_train, X_octave_test, y_target_note_train, y_target_note_test, y_target_duration_train, y_target_duration_test, y_target_step_train, y_target_step_test, y_target_octave_train, y_target_octave_test

    ### 3. Treino do modelo de classificação
    def trainModel(self):
        L_symb_notes, X_note_train, X_note_test, X_duration_train, X_duration_test, X_step_train, X_step_test, X_octave_train, X_octave_test, y_target_note_train, y_target_note_test, y_target_duration_train, y_target_duration_test, y_target_step_train, y_target_step_test, y_target_octave_train, y_target_octave_test = self.prepareData()
        
        model_note = svm.SVC(kernel='linear')
        model_duration = svm.SVC(kernel='linear')
        model_step = svm.SVC(kernel='linear')
        model_octave = svm.SVC(kernel='linear')

        model_note.fit(X_note_train, y_target_note_train)
        model_duration.fit(X_duration_train, y_target_duration_train)
        model_step.fit(X_step_train, y_target_step_train)
        model_octave.fit(X_octave_train, y_target_octave_train)

        note_pred = model_note.predict(X_note_test)
        duration_pred = model_duration.predict(X_duration_test)
        step_pred = model_step.predict(X_step_test)
        octave_pred = model_octave.predict(X_octave_test)

        note_accuracy = accuracy_score(y_target_note_test, note_pred)
        print("Accuracy do modelo de nota:", note_accuracy)
        duration_accuracy = accuracy_score(y_target_duration_test, duration_pred)
        print("Accuracy do modelo de duração:", duration_accuracy)
        step_accuracy = accuracy_score(y_target_step_test, step_pred)
        print("Accuracy do modelo de passo:", step_accuracy)
        octave_accuracy = accuracy_score(y_target_octave_test, octave_pred)
        print("Accuracy do modelo de oitava:", octave_accuracy)

        return L_symb_notes, model_note, model_duration, model_step, model_octave
    
    ### 4. Geração do feedback
    def generateFeedback(self):  
        cwd = os.getcwd()
        if self.artist == "Beethoven":
            filepath = cwd + "/templates/images/beethoven/mid/" + self.filename + ".mid"
        elif self.artist == "Halsey":
            filepath = cwd + "/templates/images/halsey/mid/" + self.filename + ".mid"
        else:
            filepath = cwd + "/templates/images/madonna/mid/" + self.filename + ".mid"

        music_midi = converter.parse(filepath)
        new_music = np.array([str(n) for n in music_midi.flat.notes])

        if len(new_music) > self.length:
            new_music = new_music[:self.length]
        elif len(new_music) < self.length:
            padding = ['0'] * (self.length - len(new_music))
            new_music = np.concatenate((new_music, padding))

        L_symb_notes, model_note, model_duration, model_step, model_octave = self.trainModel()

        new_music = np.reshape(new_music, (1, self.length, 1)) / float(L_symb_notes)

        note_feedback = model_note.predict(new_music)
        duration_feedback = model_duration.predict(new_music)
        step_feedback = model_step.predict(new_music)
        octave_feedback = model_octave.predict(new_music)

        if (note_feedback < 0.5).all() and (duration_feedback < 0.5).all() and (step_feedback < 0.5).all() and (octave_feedback < 0.5).all():
            print("A música gerada é considerada boa.")
        elif (note_feedback >= 0.5).any() or (duration_feedback >= 0.5).any() or (step_feedback >= 0.5).any() or (octave_feedback >= 0.5).any():
            print("A música gerada é considerada má.")
        else:
            print("A música gerada é considerada média.")

    def run(self):
        self.importData()
        self.generateFeedback()

if __name__ == '__main__':
    model = ClassifyMusic('Beethoven', 'beeth1')
    model.run()