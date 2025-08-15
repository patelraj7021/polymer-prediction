# NeurIPS - Open Polymer Prediction 2025

*Can your model unlock the secrets of polymers? In this competition, you're tasked with predicting the fundamental properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.* 

...from the [Kaggle competition description](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview)

## Project Overview
In this competition, participants are tasked with predicting five key polymer properties based on their chemical structure.
Polymers are represented by the SMILES convention, a string of characters that defines the structure of the polymer's constituent molecules.
The target predictions are a set of five floating-point numbers.

The transformer architecture is noted for its adept capacity to capture a generalizable understanding of the information in string-based data.
Motivated by this, I chose to develop a transformer-style model to predict the required quantities.
I adapt the standard transformer architecture so that it's better suited for this problem, however, and my modifications are outlined below.

## Model Description
I only outline deviations from the classic "Attention Is All You Need" transformer architecture below, and anything not mentioned here is assumed to be the same as the original paper.

### Removal of Decoder Layers
This is the most significant change from the full encoder-decoder transformer.
For this prediction problem, I'm only looking to leverage the self-attention mechanism to enable the model to learn relationships between molecules in the polymer, which may be useful in predicting the target properties.
I don't need to predict the next character in a SMILES string and, in general, I don't need to convert the rich higher-dimensional representations of SMILES strings back to characters.
While it's possible to train an encoder-decoder transformer with SMILES data and then only use the encoder half, I hypothesized that it would be more effective to train the model to learn representations that are useful strictly in the context of predicting the relevant properties.
The encoder-only transformer outputs its learned representation of SMILES strings to a fully-connected feedforward network, which then outputs the five target predictions.

### Masking
There is no masking of SMILES characters in this model.
While other encoder-only models like the BERT family mask random tokens and train to predict them, this model isn't training to do that.
It's training to predict polymer properties directly, and the full context of all molecules in the polymer would be best for doing so.


### Relative Positional Encoding?

### CNN in between transformer and FFN?
