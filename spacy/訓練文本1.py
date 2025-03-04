from random import shuffle as rndmShuffle
NLP=spcyLoad('en_ner_bionlp13cg_md')
if 'ner' not in nlp.pipe_names:
	ner=NLP.create_pipe('ner')
	NLP.add_pipe(ner, last=True)
else: ner=NLP.get_pipe('ner')

otherPipes = [pname for pname in NLP.pipe_names if pname != 'ner']

with NLP.disable_pipes(*otherPipes):
	optimizer = NLP.begin_training()
	epsilon, nerLosses=1E-6, 1
	while nerLosses>epsilon:
		print("Statring iteration " + nerLosses)
		rndmShuffle(TRAIN_DATA)
		losses={}
		for text, annotations in TRAIN_DATA:
			NLP.update([text], [annotations], drop=.2, sgd=optimizer, losses=losses)
		nerLosses=losses.get('ner')
	return NLP
