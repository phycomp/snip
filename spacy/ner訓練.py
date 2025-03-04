#!/usr/bin/env python
# coding: utf-8
# Load Packages
from __future__ import unicode_literals, print_function

import plac #  wrapper over argparse
import random
from pathlib import Path
import spacy
from tqdm import tqdm # loading bar

nlp1 = spacy.load('en')

docx1 = nlp1(u"Who was Kofi Annan?")

for token in docx1.ents:
    print(token.text,token.start_char, token.end_char,token.label_)

docx2 = nlp1(u"Who is Steve Jobs?")

for token in docx2.ents:
    print(token.text,token.start_char, token.end_char,token.label_)

docx3 = nlp1(u"Who is Shaka Khan?")
# training data
TRAIN_DATA = [
    ('Who is Kofi Annan?', {
        'entities': [(8, 18, 'PERSON')]
    }),
     ('Who is Steve Jobs?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]

## plac is wrapper for argparser 
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("C:\Users\This PC\Documents\JLabs\JFlow", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

# Define our variables
model = None
output_dir=Path("C:\\Users\\This PC\\Documents\\JLabs\\JFlow")
n_iter=100


# #### Load the model

if model is not None:
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")


# #### Set Up the Pipeline

# create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe('ner')


# #### Train the Recognizer
# + Add labels,Annotate them
# + Pipes
# + Begin_training()

    # add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

    # get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)


# #### Test the trained model

# test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# #### Save the Model

# save model to output directory
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

# #### Test The Saved Model
# + NB Output Directory

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# ### Adding Additional Entity Types
# ### Natural Language Processing With SpaCy
# ![title](SpaCy_logo.png)

# #### Training the Named Entity Recognizer (NER)
# ##### Adding An Additional Entity (NER)
# + Load the model
#     + spacy.load('en')
#      - Disable existing pipe line (nlp.disable_pipes)
#     + spacy.blank('en')
#      - Added Entity Recognizer to Pipeline
# + Add a Label eg(ner.add_label(LABEL) & (nlp.begin_training())
# + Shuffle and loop over the examples
#      - update the model (nlp.update)
# + Save the trained model (nlp.to_disk)
# + Test

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


# new entity label
LABEL = 'ANIMAL'

TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, 'ANIMAL')]
    }),

    ("horses?", {
        'entities': [(0, 6, 'ANIMAL')]
    })
]


# In[82]:



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))


def main(model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'Do you like horses?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


# if __name__ == '__main__':
#     plac.call(main)
main()
