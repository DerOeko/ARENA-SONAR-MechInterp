
#%%
# Define word lists (expand these for more variety!)
articles = ["the", "a", "one", "some", "any"]
adjectives = ["big", "small", "red", "blue", "green", "happy", "sad", "quick", "slow", "bright"]
nouns_singular = ["dog", "cat", "house", "tree", "car", "book", "computer", "scientist", "idea", "river"]
nouns_plural = ["dogs", "cats", "houses", "trees", "cars", "books", "computers", "scientists", "ideas", "rivers"]
# Simple present tense verbs
verbs_singular_intransitive = ["runs", "sleeps", "jumps", "flies", "swims", "thinks", "works"] # Verb for singular subject, no object
verbs_plural_intransitive = ["run", "sleep", "jump", "fly", "swim", "think", "work"]       # Verb for plural subject, no object
verbs_singular_transitive = ["chases", "sees", "likes", "builds", "writes", "reads", "finds"] # Verb for singular subject, needs object
verbs_plural_transitive = ["chase", "see", "like", "build", "write", "read", "find"]     # Verb for plural subject, needs object
prepositions = ["on", "in", "under", "near", "with", "from", "to", "above"]
conjunctions = ["and", "but", "or", "so"] # For more complex sentences later if needed

# Sentence Patterns:
# S = Subject, V = Verb, O = Object, Adj = Adjective, Art = Article, PrepP = Prepositional Phrase
# For simplicity, we'll focus on a few basic ones.

# Pattern 1: Art - (Adj) - Noun_s - Verb_s_intransitive
# Pattern 2: Art - (Adj) - Noun_p - Verb_p_intransitive
# Pattern 3: Art - (Adj) - Noun_s - Verb_s_transitive - Art - (Adj) - Noun_s
# Pattern 4: Art - (Adj) - Noun_p - Verb_p_transitive - Art - (Adj) - Noun_p

# (Adj) indicates optional adjective

import itertools

generated_sentences = set()
num_sentences_to_generate = 3000 # Your target

# Helper to capitalize
def capitalize_first(s):
    return s[0].upper() + s[1:]

# --- Generate for Pattern 1: Art - Noun_s - Verb_s_intransitive ---
print("Generating for Pattern 1...")
slots_p1 = [articles, nouns_singular, verbs_singular_intransitive]
for combo in itertools.product(*slots_p1):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()


# --- Generate for Pattern 1b: Art - Adj - Noun_s - Verb_s_intransitive ---
print("Generating for Pattern 1b...")
slots_p1b = [articles, adjectives, nouns_singular, verbs_singular_intransitive]
for combo in itertools.product(*slots_p1b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]} {combo[3]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()


# --- Generate for Pattern 2: Art - Noun_p - Verb_p_intransitive ---
print("Generating for Pattern 2...")
slots_p2 = [articles, nouns_plural, verbs_plural_intransitive]
for combo in itertools.product(*slots_p2):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()

# --- Generate for Pattern 2b: Art - Adj - Noun_p - Verb_p_intransitive ---
print("Generating for Pattern 2b...")
slots_p2b = [articles, adjectives, nouns_plural, verbs_plural_intransitive]
for combo in itertools.product(*slots_p2b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]} {combo[3]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()


# --- Generate for Pattern 3: Art - Noun_s - Verb_s_transitive - Art - Noun_s ---
print("Generating for Pattern 3...")
slots_p3 = [articles, nouns_singular, verbs_singular_transitive, articles, nouns_singular]
for combo in itertools.product(*slots_p3):
    if len(generated_sentences) >= num_sentences_to_generate: break
    # Avoid subject and object being the same for more sense
    if combo[1] == combo[4]: continue
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]} {combo[3]} {combo[4]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()


# --- Generate for Pattern 3b: Art - Adj - Noun_s - Verb_s_transitive - Art - Adj - Noun_s ---
print("Generating for Pattern 3b...")
slots_p3b = [articles, adjectives, nouns_singular, verbs_singular_transitive, articles, adjectives, nouns_singular]
for combo in itertools.product(*slots_p3b):
    if len(generated_sentences) >= num_sentences_to_generate: break
    if combo[2] == combo[6] and combo[1] == combo[5]: continue # Avoid "The big dog chases the big dog."
    sentence = f"{capitalize_first(combo[0])} {combo[1]} {combo[2]} {combo[3]} {combo[4]} {combo[5]} {combo[6]}."
    generated_sentences.add(sentence)
if len(generated_sentences) >= num_sentences_to_generate: print("Target reached."); exit()

# Add more patterns for plural objects, prepositional phrases etc. to get more variety.

final_sentences_list = list(generated_sentences)

print(f"\nGenerated {len(final_sentences_list)} unique sentences.")
print("First 20 examples:")
for i, s in enumerate(final_sentences_list[:20]):
    print(f"{i+1}. {s}")

# Save to a file
output_filename = "itertools_generated_sentences.txt"
with open(output_filename, "w") as f:
    for sentence in final_sentences_list:
        f.write(sentence + "\n")
print(f"\nSaved {len(final_sentences_list)} sentences to {output_filename}")

# If you didn't reach 3000, you need to add more words to your lists or define more sentence patterns.
if len(final_sentences_list) < num_sentences_to_generate:
    print(f"\nWarning: Could only generate {len(final_sentences_list)} unique sentences with the current vocabulary and patterns.")
    print("Consider expanding your word lists or adding more sentence patterns to reach the target of 3000.")
# %%
