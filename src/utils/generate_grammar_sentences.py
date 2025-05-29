import itertools
import random

# Define word lists (expand these for more variety!)
articles = ["the", "a", "one", "some", "any"]
adjectives = ["big", "small", "red", "blue", "green", "happy", "sad", "quick", "slow", "bright"]
nouns_singular = ["dog", "cat", "house", "tree", "car", "book", "computer", "scientist", "idea", "river"]
nouns_plural = ["dogs", "cats", "houses", "trees", "cars", "books", "computers", "scientists", "ideas", "rivers"]
verbs_singular_intransitive = ["runs", "sleeps", "jumps", "flies", "swims", "thinks", "works"]
verbs_plural_intransitive = ["run", "sleep", "jump", "fly", "swim", "think", "work"]
verbs_singular_transitive = ["chases", "sees", "likes", "builds", "writes", "reads", "finds"]
verbs_plural_transitive = ["chase", "see", "like", "build", "write", "read", "find"]
prepositions = ["on", "in", "under", "near", "with", "from", "to", "above"]
conjunctions = ["and", "but", "or", "so"]

# German word lists (simplified for demonstration, gender/case are complex!)
# Note: For accurate German, you'd need to handle gender, case, and verb conjugations rigorously.
# This is a very basic mapping.
de_articles = {
    "the": {"m": "der", "f": "die", "n": "das", "pl": "die"},
    "a": {"m": "ein", "f": "eine", "n": "ein"}
}
de_adjectives = {
    "big": "groß", "small": "klein", "red": "rot", "blue": "blau", "green": "grün",
    "happy": "glücklich", "sad": "traurig", "quick": "schnell", "slow": "langsam", "bright": "hell"
}
de_nouns_singular = {
    "dog": {"word": "Hund", "gender": "m"}, "cat": {"word": "Katze", "gender": "f"},
    "house": {"word": "Haus", "gender": "n"}, "tree": {"word": "Baum", "gender": "m"},
    "car": {"word": "Auto", "gender": "n"}, "book": {"word": "Buch", "gender": "n"},
    "computer": {"word": "Computer", "gender": "m"}, "scientist": {"word": "Wissenschaftler", "gender": "m"},
    "idea": {"word": "Idee", "gender": "f"}, "river": {"word": "Fluss", "gender": "m"}
}
de_nouns_plural = {
    "dogs": "Hunde", "cats": "Katzen", "houses": "Häuser", "trees": "Bäume",
    "cars": "Autos", "books": "Bücher", "computers": "Computer", "scientists": "Wissenschaftler",
    "ideas": "Ideen", "rivers": "Flüsse"
}
de_verbs_singular_intransitive = {
    "runs": "läuft", "sleeps": "schläft", "jumps": "springt", "flies": "fliegt",
    "swims": "schwimmt", "thinks": "denkt", "works": "arbeitet"
}
de_verbs_plural_intransitive = {
    "run": "laufen", "sleep": "schlafen", "jump": "springen", "fly": "fliegen",
    "swim": "schwimmen", "think": "denken", "work": "arbeiten"
}
de_verbs_singular_transitive = {
    "chases": "jagt", "sees": "sieht", "likes": "mag", "builds": "baut",
    "writes": "schreibt", "reads": "liest", "finds": "findet"
}
de_verbs_plural_transitive = {
    "chase": "jagen", "see": "sehen", "like": "mögen", "build": "bauen",
    "write": "schreiben", "read": "lesen", "find": "finden"
}


generated_sentences = set()
generated_german_sentences = set()
generated_ungrammatical_sentences = set()
num_sentences_to_generate = 3000 # Your target

# Helper to capitalize
def capitalize_first(s):
    return s[0].upper() + s[1:]

# Helper for German translation (very basic)
def translate_to_german(english_sentence_parts, pattern_type):
    try:
        if pattern_type == "intransitive_singular":
            # Art - (Adj) - Noun_s - Verb_s_intransitive
            en_article = english_sentence_parts[0].lower()
            en_adj = english_sentence_parts[1].lower() if len(english_sentence_parts) == 4 else None
            en_noun = english_sentence_parts[2].lower() if len(english_sentence_parts) == 4 else english_sentence_parts[1].lower()
            en_verb = english_sentence_parts[3].lower() if len(english_sentence_parts) == 4 else english_sentence_parts[2].lower()

            noun_info = de_nouns_singular.get(en_noun)
            if not noun_info: return None
            de_noun = noun_info["word"]
            de_gender = noun_info["gender"]

            de_art = de_articles.get(en_article, {}).get(de_gender)
            if not de_art: return None

            de_verb = de_verbs_singular_intransitive.get(en_verb)
            if not de_verb: return None

            if en_adj:
                de_adj = de_adjectives.get(en_adj)
                if not de_adj: return None
                return f"{capitalize_first(de_art)} {de_adj} {de_noun} {de_verb}."
            else:
                return f"{capitalize_first(de_art)} {de_noun} {de_verb}."

        elif pattern_type == "intransitive_plural":
            # Art - (Adj) - Noun_p - Verb_p_intransitive
            en_article = english_sentence_parts[0].lower()
            en_adj = english_sentence_parts[1].lower() if len(english_sentence_parts) == 4 else None
            en_noun = english_sentence_parts[2].lower() if len(english_sentence_parts) == 4 else english_sentence_parts[1].lower()
            en_verb = english_sentence_parts[3].lower() if len(english_sentence_parts) == 4 else english_sentence_parts[2].lower()

            de_art = de_articles.get(en_article, {}).get("pl")
            if not de_art: return None
            de_noun = de_nouns_plural.get(en_noun)
            if not de_noun: return None
            de_verb = de_verbs_plural_intransitive.get(en_verb)
            if not de_verb: return None

            if en_adj:
                de_adj = de_adjectives.get(en_adj)
                if not de_adj: return None
                return f"{capitalize_first(de_art)} {de_adj} {de_noun} {de_verb}."
            else:
                return f"{capitalize_first(de_art)} {de_noun} {de_verb}."

        elif pattern_type == "transitive_singular":
            # Art - (Adj) - Noun_s - Verb_s_transitive - Art - (Adj) - Noun_s (Subject and Object)
            if len(english_sentence_parts) == 7: # With adjectives
                en_subj_art = english_sentence_parts[0].lower()
                en_subj_adj = english_sentence_parts[1].lower()
                en_subj_noun = english_sentence_parts[2].lower()
                en_verb = english_sentence_parts[3].lower()
                en_obj_art = english_sentence_parts[4].lower()
                en_obj_adj = english_sentence_parts[5].lower()
                en_obj_noun = english_sentence_parts[6].lower()

                subj_noun_info = de_nouns_singular.get(en_subj_noun)
                obj_noun_info = de_nouns_singular.get(en_obj_noun)
                if not subj_noun_info or not obj_noun_info: return None

                de_subj_noun = subj_noun_info["word"]
                de_subj_gender = subj_noun_info["gender"]
                de_obj_noun = obj_noun_info["word"]
                de_obj_gender = obj_noun_info["gender"]

                de_subj_art = de_articles.get(en_subj_art, {}).get(de_subj_gender)
                # Object in accusative case for transitive verbs, but we're simplifying.
                # For this example, we'll just use the nominative articles for objects too.
                de_obj_art = de_articles.get(en_obj_art, {}).get(de_obj_gender)
                if not de_subj_art or not de_obj_art: return None

                de_verb = de_verbs_singular_transitive.get(en_verb)
                if not de_verb: return None

                de_subj_adj = de_adjectives.get(en_subj_adj)
                de_obj_adj = de_adjectives.get(en_obj_adj)
                if not de_subj_adj or not de_obj_adj: return None

                # Simplified German sentence structure (SVO generally, but verb often second)
                # Article, adjective, noun, verb, article, adjective, noun.
                return f"{capitalize_first(de_subj_art)} {de_subj_adj} {de_subj_noun} {de_verb} {de_obj_art} {de_obj_adj} {de_obj_noun}."
            elif len(english_sentence_parts) == 5: # Without adjectives
                en_subj_art = english_sentence_parts[0].lower()
                en_subj_noun = english_sentence_parts[1].lower()
                en_verb = english_sentence_parts[2].lower()
                en_obj_art = english_sentence_parts[3].lower()
                en_obj_noun = english_sentence_parts[4].lower()

                subj_noun_info = de_nouns_singular.get(en_subj_noun)
                obj_noun_info = de_nouns_singular.get(en_obj_noun)
                if not subj_noun_info or not obj_noun_info: return None

                de_subj_noun = subj_noun_info["word"]
                de_subj_gender = subj_noun_info["gender"]
                de_obj_noun = obj_noun_info["word"]
                de_obj_gender = obj_noun_info["gender"]

                de_subj_art = de_articles.get(en_subj_art, {}).get(de_subj_gender)
                de_obj_art = de_articles.get(en_obj_art, {}).get(de_obj_gender)
                if not de_subj_art or not de_obj_art: return None

                de_verb = de_verbs_singular_transitive.get(en_verb)
                if not de_verb: return None

                return f"{capitalize_first(de_subj_art)} {de_subj_noun} {de_verb} {de_obj_art} {de_obj_noun}."
        elif pattern_type == "transitive_plural":
            # Art - (Adj) - Noun_p - Verb_p_transitive - Art - (Adj) - Noun_p (Subject and Object)
            if len(english_sentence_parts) == 7: # With adjectives
                en_subj_art = english_sentence_parts[0].lower()
                en_subj_adj = english_sentence_parts[1].lower()
                en_subj_noun = english_sentence_parts[2].lower()
                en_verb = english_sentence_parts[3].lower()
                en_obj_art = english_sentence_parts[4].lower()
                en_obj_adj = english_sentence_parts[5].lower()
                en_obj_noun = english_sentence_parts[6].lower()

                de_subj_art = de_articles.get(en_subj_art, {}).get("pl")
                de_obj_art = de_articles.get(en_obj_art, {}).get("pl")
                if not de_subj_art or not de_obj_art: return None

                de_subj_noun = de_nouns_plural.get(en_subj_noun)
                de_obj_noun = de_nouns_plural.get(en_obj_noun)
                if not de_subj_noun or not de_obj_noun: return None

                de_verb = de_verbs_plural_transitive.get(en_verb)
                if not de_verb: return None

                de_subj_adj = de_adjectives.get(en_subj_adj)
                de_obj_adj = de_adjectives.get(en_obj_adj)
                if not de_subj_adj or not de_obj_adj: return None
                return f"{capitalize_first(de_subj_art)} {de_subj_adj} {de_subj_noun} {de_verb} {de_obj_art} {de_obj_adj} {de_obj_noun}."
            elif len(english_sentence_parts) == 5: # Without adjectives
                en_subj_art = english_sentence_parts[0].lower()
                en_subj_noun = english_sentence_parts[1].lower()
                en_verb = english_sentence_parts[2].lower()
                en_obj_art = english_sentence_parts[3].lower()
                en_obj_noun = english_sentence_parts[4].lower()

                de_subj_art = de_articles.get(en_subj_art, {}).get("pl")
                de_obj_art = de_articles.get(en_obj_art, {}).get("pl")
                if not de_subj_art or not de_obj_art: return None

                de_subj_noun = de_nouns_plural.get(en_subj_noun)
                de_obj_noun = de_nouns_plural.get(en_obj_noun)
                if not de_subj_noun or not de_obj_noun: return None

                de_verb = de_verbs_plural_transitive.get(en_verb)
                if not de_verb: return None
                return f"{capitalize_first(de_subj_art)} {de_subj_noun} {de_verb} {de_obj_art} {de_obj_noun}."
    except Exception as e:
        # print(f"Error translating: {e} with parts {english_sentence_parts}")
        return None
    return None

# --- Generate for Pattern 1: Art - Noun_s - Verb_s_intransitive ---
print("Generating for Pattern 1...")
slots_p1 = [articles, nouns_singular, verbs_singular_intransitive]
for combo in itertools.product(*slots_p1):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "intransitive_singular")
    if german_sentence:
        generated_german_sentences.add(german_sentence)

# --- Generate for Pattern 1b: Art - Adj - Noun_s - Verb_s_intransitive ---
print("Generating for Pattern 1b...")
slots_p1b = [articles, adjectives, nouns_singular, verbs_singular_intransitive]
for combo in itertools.product(*slots_p1b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "intransitive_singular")
    if german_sentence:
        generated_german_sentences.add(german_sentence)


# --- Generate for Pattern 2: Art - Noun_p - Verb_p_intransitive ---
print("Generating for Pattern 2...")
slots_p2 = [articles, nouns_plural, verbs_plural_intransitive]
for combo in itertools.product(*slots_p2):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "intransitive_plural")
    if german_sentence:
        generated_german_sentences.add(german_sentence)

# --- Generate for Pattern 2b: Art - Adj - Noun_p - Verb_p_intransitive ---
print("Generating for Pattern 2b...")
slots_p2b = [articles, adjectives, nouns_plural, verbs_plural_intransitive]
for combo in itertools.product(*slots_p2b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "intransitive_plural")
    if german_sentence:
        generated_german_sentences.add(german_sentence)


# --- Generate for Pattern 3: Art - Noun_s - Verb_s_transitive - Art - Noun_s ---
print("Generating for Pattern 3...")
slots_p3 = [articles, nouns_singular, verbs_singular_transitive, articles, nouns_singular]
for combo in itertools.product(*slots_p3):
    if len(generated_sentences) >= num_sentences_to_generate: break
    # Avoid subject and object being the same for more sense
    if combo[1] == combo[4]: continue
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]} {sentence_parts[4]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "transitive_singular")
    if german_sentence:
        generated_german_sentences.add(german_sentence)


# --- Generate for Pattern 3b: Art - Adj - Noun_s - Verb_s_transitive - Art - Adj - Noun_s ---
print("Generating for Pattern 3b...")
slots_p3b = [articles, adjectives, nouns_singular, verbs_singular_transitive, articles, adjectives, nouns_singular]
for combo in itertools.product(*slots_p3b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    if combo[2] == combo[6] and combo[1] == combo[5]: continue # Avoid "The big dog chases the big dog."
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]} {sentence_parts[4]} {sentence_parts[5]} {sentence_parts[6]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "transitive_singular")
    if german_sentence:
        generated_german_sentences.add(german_sentence)

# --- Generate for Pattern 4: Art - Noun_p - Verb_p_transitive - Art - Noun_p ---
print("Generating for Pattern 4...")
slots_p4 = [articles, nouns_plural, verbs_plural_transitive, articles, nouns_plural]
for combo in itertools.product(*slots_p4):
    if len(generated_sentences) >= num_sentences_to_generate: break
    if combo[1] == combo[4]: continue
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]} {sentence_parts[4]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "transitive_plural")
    if german_sentence:
        generated_german_sentences.add(german_sentence)

# --- Generate for Pattern 4b: Art - Adj - Noun_p - Verb_p_transitive - Art - Adj - Noun_p ---
print("Generating for Pattern 4b...")
slots_p4b = [articles, adjectives, nouns_plural, verbs_plural_transitive, articles, adjectives, nouns_plural]
for combo in itertools.product(*slots_p4b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    if combo[2] == combo[6] and combo[1] == combo[5]: continue
    sentence_parts = list(combo)
    sentence = f"{capitalize_first(sentence_parts[0])} {sentence_parts[1]} {sentence_parts[2]} {sentence_parts[3]} {sentence_parts[4]} {sentence_parts[5]} {sentence_parts[6]}."
    generated_sentences.add(sentence)

    german_sentence = translate_to_german(sentence_parts, "transitive_plural")
    if german_sentence:
        generated_german_sentences.add(german_sentence)


# --- Generate Ungrammatical Sentences ---
# Strategy: Introduce specific grammatical errors by scrambling parts of the patterns.
# For example, subject-verb agreement errors, wrong word order, or incorrect part of speech placement.

print("\nGenerating Ungrammatical Sentences...")

# Ungrammatical Pattern 1: Subject-Verb Agreement (Singular Noun with Plural Verb)
# Art - (Adj) - Noun_s - Verb_p_intransitive
for _ in range(num_sentences_to_generate // 6): # Generate a fraction of the total
    art = random.choice(articles)
    adj = random.choice(adjectives)
    noun_s = random.choice(nouns_singular)
    verb_p_intransitive = random.choice(verbs_plural_intransitive)
    
    # Optional adjective
    if random.random() < 0.5: # 50% chance of including adjective
        ungrammatical_sentence = f"{capitalize_first(art)} {adj} {noun_s} {verb_p_intransitive}."
    else:
        ungrammatical_sentence = f"{capitalize_first(art)} {noun_s} {verb_p_intransitive}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)

# Ungrammatical Pattern 2: Subject-Verb Agreement (Plural Noun with Singular Verb)
# Art - (Adj) - Noun_p - Verb_s_intransitive
for _ in range(num_sentences_to_generate // 6):
    art = random.choice(articles)
    adj = random.choice(adjectives)
    noun_p = random.choice(nouns_plural)
    verb_s_intransitive = random.choice(verbs_singular_intransitive)

    if random.random() < 0.5:
        ungrammatical_sentence = f"{capitalize_first(art)} {adj} {noun_p} {verb_s_intransitive}."
    else:
        ungrammatical_sentence = f"{capitalize_first(art)} {noun_p} {verb_s_intransitive}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)

# Ungrammatical Pattern 3: Incorrect word order (Verb before Subject)
# Verb - Art - Noun - (Adj) - .
for _ in range(num_sentences_to_generate // 6):
    verb = random.choice(verbs_singular_intransitive + verbs_plural_intransitive)
    art = random.choice(articles)
    noun = random.choice(nouns_singular + nouns_plural)
    adj = random.choice(adjectives)

    if random.random() < 0.5:
        ungrammatical_sentence = f"{capitalize_first(verb)} {art} {adj} {noun}."
    else:
        ungrammatical_sentence = f"{capitalize_first(verb)} {art} {noun}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)

# Ungrammatical Pattern 4: Adjective after noun (e.g., "dog big")
# Art - Noun - Adj - Verb (for intransitive) or Art - Noun - Adj - Verb - Art - Noun - Adj (for transitive)
for _ in range(num_sentences_to_generate // 6):
    art = random.choice(articles)
    noun_s = random.choice(nouns_singular)
    adj = random.choice(adjectives)
    verb_s_intransitive = random.choice(verbs_singular_intransitive)
    ungrammatical_sentence = f"{capitalize_first(art)} {noun_s} {adj} {verb_s_intransitive}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)

for _ in range(num_sentences_to_generate // 6):
    art_subj = random.choice(articles)
    noun_s_subj = random.choice(nouns_singular)
    adj_subj = random.choice(adjectives)
    verb_s_transitive = random.choice(verbs_singular_transitive)
    art_obj = random.choice(articles)
    noun_s_obj = random.choice(nouns_singular)
    adj_obj = random.choice(adjectives)

    if noun_s_subj == noun_s_obj and adj_subj == adj_obj: continue # Avoid "The dog big chases the dog big."

    ungrammatical_sentence = f"{capitalize_first(art_subj)} {noun_s_subj} {adj_subj} {verb_s_transitive} {art_obj} {noun_s_obj} {adj_obj}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)


# Ungrammatical Pattern 5: Missing article
# (Adj) - Noun_s - Verb_s_intransitive
for _ in range(num_sentences_to_generate // 6):
    adj = random.choice(adjectives)
    noun_s = random.choice(nouns_singular)
    verb_s_intransitive = random.choice(verbs_singular_intransitive)

    if random.random() < 0.5:
        ungrammatical_sentence = f"{capitalize_first(adj)} {noun_s} {verb_s_intransitive}."
    else:
        ungrammatical_sentence = f"{capitalize_first(noun_s)} {verb_s_intransitive}."
    generated_ungrammatical_sentences.add(ungrammatical_sentence)


final_sentences_list = list(generated_sentences)
final_german_sentences_list = list(generated_german_sentences)
final_ungrammatical_sentences_list = list(generated_ungrammatical_sentences)

print(f"\nGenerated {len(final_sentences_list)} unique grammatical English sentences.")
print(f"Generated {len(final_german_sentences_list)} unique German sentences (basic translation).")
print(f"Generated {len(final_ungrammatical_sentences_list)} unique ungrammatical English sentences.")

print("\n--- First 10 Grammatical English Examples ---")
for i, s in enumerate(final_sentences_list[:10]):
    print(f"{i+1}. {s}")

print("\n--- First 10 German Translation Examples ---")
for i, s in enumerate(final_german_sentences_list[:10]):
    print(f"{i+1}. {s}")

print("\n--- First 10 Ungrammatical English Examples ---")
for i, s in enumerate(final_ungrammatical_sentences_list[:10]):
    print(f"{i+1}. {s}")

# Save to files
output_grammatical_en_filename = "grammatical_english_sentences.txt"
with open(output_grammatical_en_filename, "w") as f:
    for sentence in final_sentences_list:
        f.write(sentence + "\n")
print(f"\nSaved {len(final_sentences_list)} grammatical English sentences to {output_grammatical_en_filename}")

output_german_filename = "german_sentences.txt"
with open(output_german_filename, "w") as f:
    for sentence in final_german_sentences_list:
        f.write(sentence + "\n")
print(f"Saved {len(final_german_sentences_list)} German sentences to {output_german_filename}")

output_ungrammatical_en_filename = "ungrammatical_english_sentences.txt"
with open(output_ungrammatical_en_filename, "w") as f:
    for sentence in final_ungrammatical_sentences_list:
        f.write(sentence + "\n")
print(f"Saved {len(final_ungrammatical_sentences_list)} ungrammatical English sentences to {output_ungrammatical_en_filename}")

if len(final_sentences_list) < num_sentences_to_generate:
    print(f"\nWarning: Could only generate {len(final_sentences_list)} unique grammatical English sentences with the current vocabulary and patterns.")
    print("Consider expanding your English word lists or adding more sentence patterns to reach the target of 3000.")