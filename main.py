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
    # 总体说明：对传入的旋律（melody）计算一个综合分数，分数由多个音乐学特征的子评分构成。
    # 不改变旋律结构或时值，只计算并返回一个数值（同时赋值到 melody.fitness）。
    score = 0.0
    notes = melody.notes
    if not notes:
        return 0.0

    # === 1. 音阶遵循（C大调） ===
    # 说明：检查每个音符的音高模12是否属于C大调音阶（C D E F G A B），
    # 按比例（属调内音符数 / 总音符数）给最多40分的奖励。
    c_major_indices = {0, 2, 4, 5, 7, 9, 11}
    in_key_count = sum(1 for n in notes if (n.pitch % 12) in c_major_indices)
    score += (in_key_count / len(notes)) * 40  # Max 40 points

    # === 2. 旋律音程（Intervals）评分调整 ===
    # 说明：优先平滑的阶进（小二/大二、三度等），对大跳和超大跳进行扣分。
    #       重复音给予少量正分，步进给予较高分，八度视为可接受。
    #       最终将该区间分数裁剪到0~40的范围后加入总分。
    interval_score = 0
    for i in range(len(notes) - 1):
        diff = abs(notes[i].pitch - notes[i+1].pitch)
        if diff == 0:
            interval_score += 1 # 重复音给1分
        elif diff <= 2: # 小二/大二（步进）给2分
            interval_score += 2
        elif diff <= 4: # 三度给1分
            interval_score += 1
        elif diff == 12: # 八度视为可接受，给1分
            interval_score += 1
        elif diff > 7: # 大跳（5度以上）扣分
            interval_score -= 2
        if diff > 12: # 极大跳（超八度）额外重扣
            interval_score -= 5

    # 将 interval_score 限制在合理范围后加入总分（防止极端值影响太大）
    score += max(0, min(40, interval_score)) 

    # === 3. 终止音（Ending Note）偏好 ===
    # 说明：结尾音偏好落在 C(主音) 或 G(属音)，分别给予不同的奖励，E（中音）次之。
    last_note = notes[-1].pitch % 12
    if last_note == 0: # C
        score += 15
    elif last_note == 7: # G
        score += 10
    elif last_note == 4: # E
        score += 5

    # === 4. 中音域偏好（Mid-range preference） ===
    # 说明：鼓励音高靠近允许范围中心的位置（更稳定、听感中性），按偏离程度线性评分，最多30分。
    center = (PITCH_MIN + PITCH_MAX) / 2.0
    half_range = (PITCH_MAX - PITCH_MIN) / 2.0 if (PITCH_MAX - PITCH_MIN) > 0 else 1.0
    center_sum = 0.0
    for n in notes:
        norm = min(1.0, abs(n.pitch - center) / half_range)
        center_sum += (1.0 - norm)
    center_score = (center_sum / len(notes)) * 30.0  # up to 30 points
    score += center_score

    # === 5. 连续上行/下行跑动的惩罚（防止单向长程移动） ===
    # 说明：如果旋律连续多步单向下行或上行，会降低可听性（单调），对过长的下行/上行段落进行递增惩罚。
    desc_run = 0
    asc_run = 0
    for i in range(len(notes) - 1):
        if notes[i+1].pitch < notes[i].pitch:
            desc_run += 1
            asc_run = 0
        elif notes[i+1].pitch > notes[i].pitch:
            asc_run += 1
            desc_run = 0
        else:
            asc_run = 0
            desc_run = 0

        # 对较长的下行段落更强烈地扣分
        if desc_run >= 3:
            # 超过长度阈值后按递增方式扣分
            score -= 2 * (desc_run - 2)
        if asc_run >= 4:
            score -= 1 * (asc_run - 3)

    # === 6. 重复惩罚（Repetition Penalty） ===
    # 说明：连续大量重复同一音会降低音乐性（如 C C C），当连续重复超过阈值时扣分。
    consecutive = 0
    for i in range(len(notes) - 1):
        if notes[i].pitch == notes[i+1].pitch:
            consecutive += 1
            if consecutive >= 2: # 超过2次重复（例如 C C C）扣分
                score -= 5
        else:
            consecutive = 0

    # === 7. 节奏多样性（Rhythm Variety） ===
    # 说明：如果过度使用短音（八分音符 0.5），会让节奏显得拥挤，超过 50% 则按比例扣分。
    eighth_note_count = sum(1 for n in notes if n.duration == 0.5)
    if len(notes) > 0:
        eighth_ratio = eighth_note_count / len(notes)
        if eighth_ratio > 0.5: # 如果八分音符占比超过50%
            score -= (eighth_ratio - 0.5) * 20 # 按超出比例扣分

    # === 8. 起始音（Start Note）偏好 ===
    # 说明：鼓励以主音C、属音G或中音E作为开头，这里按不同权重给予小幅奖励。
    first_note_pitch = notes[0].pitch % 12
    if first_note_pitch == 0: # C
        score += 5
    elif first_note_pitch == 7: # G
        score += 3
    elif first_note_pitch == 4: # E
        score += 2

    # === 9. 节拍/动机重复（Rhythmic Structure） ===
    # 说明：检测第1小节和第3小节的节奏型是否相同（A-?-A-?）或者第1和第2小节是否相同（A-A-?-?），
    #       发现重复时给予节奏结构上的奖励。
    current_time = 0.0
    bar_rhythms = [[], [], [], []] # Durations for Bar 1, 2, 3, 4
    
    for n in notes:
        bar_idx = int(current_time // 4.0)
        if bar_idx < 4:
            bar_rhythms[bar_idx].append(n.duration)
        current_time += n.duration
        
    # 节奏型重复的奖励
    if bar_rhythms[0] and bar_rhythms[2] and bar_rhythms[0] == bar_rhythms[2]:
        score += 15 # 强烈奖励 A-?-A-? 结构
    elif bar_rhythms[0] and bar_rhythms[1] and bar_rhythms[0] == bar_rhythms[1]:
        score += 10 # 奖励 A-A-?-? 结构

    # === 10. 强拍对齐（Strong Beat Alignment） ===
    # 说明：奖励那些落在强拍（小节首位或第二强拍，如 0.0, 2.0 等）开始的音符，每个命中计分，最多加10分。
    current_time = 0.0
    on_beat_score = 0
    for n in notes:
        # 若当前时间接近偶数拍（0,2,4...）视为强拍起始
        if abs(current_time % 2.0) < 0.01:
            on_beat_score += 1
        current_time += n.duration
    score += min(10, on_beat_score) # Cap at 10 points

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
