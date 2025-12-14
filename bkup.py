import random
import copy
from audio_generator import save_melody_as_mp3

# Constants
PITCH_MIN = 53  # F3
PITCH_MAX = 79  # G5
PITCH_SET = list(range(PITCH_MIN, PITCH_MAX + 1))
DURATIONS = [0.5, 1.0, 2.0] # Eighth, Quarter, Half. (Whole note 4.0 might be too long for variety in 4 bars)
TARGET_DURATION = 16.0 # 4 bars * 4 beats (4/4 time)

# Musical Helpers
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_name(midi_num):
    octave = (midi_num // 12) - 1
    note_idx = midi_num % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"

class Note:
    def __init__(self, pitch, duration):
        self.pitch = pitch # MIDI number
        self.duration = duration # Beats (0.5, 1.0, etc.)

    def __repr__(self):
        return f"{midi_to_name(self.pitch)}({self.duration})"

class Melody:
    def __init__(self, notes=None):
        self.notes = notes if notes else []
        self.fitness = 0.0

    def calculate_total_duration(self):
        return sum(n.duration for n in self.notes)

    def __repr__(self):
        return " ".join([str(n) for n in self.notes])

def generate_random_melody():
    melody_notes = []
    current_duration = 0.0
    
    while current_duration < TARGET_DURATION:
        # Pick a random duration that fits
        remaining = TARGET_DURATION - current_duration
        possible_durations = [d for d in DURATIONS if d <= remaining]
        
        if not possible_durations:
            # If no standard duration fits (unlikely with 0.5 available), fill with 0.5
            duration = 0.5
        else:
            duration = random.choice(possible_durations)
            
        pitch = random.choice(PITCH_SET)
        melody_notes.append(Note(pitch, duration))
        current_duration += duration
        
    # Ensure exact length (floating point fix)
    if current_duration > TARGET_DURATION:
        # Trim last note or adjust
        pass 
        
    return Melody(melody_notes)

def calculate_fitness(melody):
    score = 0.0
    notes = melody.notes
    if not notes:
        return 0.0

    # 1. Scale Adherence (C Major)
    # C Major MIDI notes % 12 are: 0, 2, 4, 5, 7, 9, 11
    c_major_indices = {0, 2, 4, 5, 7, 9, 11}
    in_key_count = sum(1 for n in notes if (n.pitch % 12) in c_major_indices)
    score += (in_key_count / len(notes)) * 40  # Max 40 points

    # 2. Intervals 评分调整
    # Smooth steps are good. Large leaps are bad.
    interval_score = 0
    for i in range(len(notes) - 1):
        diff = abs(notes[i].pitch - notes[i+1].pitch)
        if diff == 0:
            interval_score += 1 # Repetition is okay, but steps are better 重复音给1分
        elif diff <= 2: # Step (1 or 2 semitones) 级进给2分
            interval_score += 2
        elif diff <= 4: # Third (3 or 4 semitones)
            interval_score += 1
        elif diff == 12: # Octave (okay)
            interval_score += 1
        elif diff > 7: # Large leap
            interval_score -= 2
        if diff > 12: # Huge leap
            interval_score -= 5
    
    # Normalize interval score roughly
    # Assuming avg 20 notes, max potential interval score is ~40.
    score += max(0, min(40, interval_score)) 

    # 3. Ending Note 期望G或C结尾
    # End on C (0) or G (7)
    last_note = notes[-1].pitch % 12
    if last_note == 0: # C
        score += 20
    elif last_note == 7: # G
        score += 10
    elif last_note == 4: # E
        score += 5

    # 4. Repetition Penalty 重复惩罚
    consecutive = 0
    for i in range(len(notes) - 1):
        if notes[i].pitch == notes[i+1].pitch:
            consecutive += 1
            if consecutive >= 2: # More than 2 same notes in a row (e.g. C C C)
                score -= 5
        else:
            consecutive = 0

    # 5. Range Penalty 太高音惩罚
    for n in notes:
        if n.pitch > 75:
            score -= 1
        
    melody.fitness = score
    return score

def split_melody_at_time(melody, split_time):
    """Splits a melody into two parts at a specific time."""
    left_notes = []
    right_notes = []
    current_time = 0.0
    
    for note in melody.notes:
        note_end = current_time + note.duration
        
        # Floating point tolerance
        if note_end <= split_time + 0.001:
            left_notes.append(copy.deepcopy(note))
        elif current_time >= split_time - 0.001:
            right_notes.append(copy.deepcopy(note))
        else:
            # Note straddles the split point
            dur1 = split_time - current_time
            dur2 = note_end - split_time
            
            if dur1 > 0.001:
                left_notes.append(Note(note.pitch, dur1))
            if dur2 > 0.001:
                right_notes.append(Note(note.pitch, dur2))
                
        current_time += note.duration
        
    return left_notes, right_notes

def crossover(p1, p2):
    # Pick a split point at bar lines: 4.0, 8.0, 12.0
    split_point = random.choice([4.0, 8.0, 12.0])
    
    p1_left, p1_right = split_melody_at_time(p1, split_point)
    p2_left, p2_right = split_melody_at_time(p2, split_point)
    
    # Create children
    c1_notes = p1_left + p2_right
    c2_notes = p2_left + p1_right
    
    return Melody(c1_notes), Melody(c2_notes)

def mutate(melody, mutation_rate=0.1):
    # Pitch Mutation
    for note in melody.notes:
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Shift +/- 1 or 2 semitones
                shift = random.choice([-2, -1, 1, 2])
                new_pitch = note.pitch + shift
                if PITCH_MIN <= new_pitch <= PITCH_MAX:
                    note.pitch = new_pitch
            else:
                # Random pitch
                note.pitch = random.choice(PITCH_SET)

    # Rhythm Mutation (Split)
    if random.random() < mutation_rate:
        if not melody.notes: return
        idx = random.randint(0, len(melody.notes) - 1)
        note = melody.notes[idx]
        if note.duration >= 1.0:
            # Split
            new_dur = note.duration / 2.0
            note.duration = new_dur
            # Insert new note with same pitch
            new_note = Note(note.pitch, new_dur)
            melody.notes.insert(idx + 1, new_note)

def transform_transpose(melody):
    shift = random.choice([-2, -1, 1, 2, 12, -12])
    new_notes = []
    for n in melody.notes:
        new_pitch = n.pitch + shift
        new_pitch = max(PITCH_MIN, min(PITCH_MAX, new_pitch))
        new_notes.append(Note(new_pitch, n.duration))
    return Melody(new_notes)

def transform_inversion(melody):
    if not melody.notes: return Melody([])
    center = melody.notes[0].pitch
    new_notes = []
    for n in melody.notes:
        diff = n.pitch - center
        new_pitch = center - diff
        new_pitch = max(PITCH_MIN, min(PITCH_MAX, new_pitch))
        new_notes.append(Note(new_pitch, n.duration))
    return Melody(new_notes)

def transform_retrograde(melody):
    new_notes = [copy.deepcopy(n) for n in reversed(melody.notes)]
    return Melody(new_notes)

def initialize_population(size=20):
    pop = [generate_random_melody() for _ in range(size)]
    for ind in pop:
        calculate_fitness(ind)
    return pop

def tournament_selection(population, k=3):
    candidates = random.sample(population, k)
    candidates.sort(key=lambda x: x.fitness, reverse=True)
    return candidates[0]

def run_genetic_algorithm():
    POP_SIZE = 20
    GENERATIONS = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    TRANSFORM_RATE = 0.1

    population = initialize_population(POP_SIZE)
    
    print(f"Initial Best Fitness: {max(ind.fitness for ind in population):.2f}")

    for gen in range(GENERATIONS):
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elitism: Keep top 2
        new_population.append(copy.deepcopy(population[0]))
        new_population.append(copy.deepcopy(population[1]))
        
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            mutate(child1, MUTATION_RATE)
            mutate(child2, MUTATION_RATE)
            
            # Transformations
            if random.random() < TRANSFORM_RATE:
                op = random.choice([transform_transpose, transform_inversion, transform_retrograde])
                child1 = op(child1)
            if random.random() < TRANSFORM_RATE:
                op = random.choice([transform_transpose, transform_inversion, transform_retrograde])
                child2 = op(child2)
                
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)
        
        population = new_population
        
        # Recalculate fitness
        for ind in population:
            calculate_fitness(ind)
            
        best_fitness = max(ind.fitness for ind in population)
        if (gen + 1) % 10 == 0:
            print(f"Generation {gen+1}: Best Fitness = {best_fitness:.2f}")

    # Final Result
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_melody = population[0]
    print("\nBest Melody Found:")
    print(best_melody)
    print(f"Fitness: {best_melody.fitness:.2f}")
    print(f"Total Duration: {best_melody.calculate_total_duration()}")
    
    # Generate MP3
    save_melody_as_mp3(best_melody, "best_melody.mp3")

if __name__ == "__main__":
    run_genetic_algorithm()
