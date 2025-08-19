# NeurIPS - Open Polymer Prediction 2025

*Can your model unlock the secrets of polymers? In this competition, you're tasked with predicting the fundamental properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.* 

...from the [Kaggle competition description](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview)

## Project Overview
In this competition, participants are tasked with predicting five key polymer properties based on their chemical structure.
Polymers are represented by the SMILES convention, a string of characters that defines the structure of the polymer's constituent molecules.
The target predictions are a set of five floating-point numbers.

The transformer architecture is noted for its capacity to capture a generalizable understanding of the information in string-based data.
Motivated by this, I chose to develop a transformer-style model to predict the required quantities.
I adapt the standard transformer architecture so that it's better suited for this problem, however, and my modifications are outlined below.

## Model Description
I only outline deviations from the classic "Attention Is All You Need" transformer architecture below, and anything not mentioned here is assumed to be the same as the original paper.

**I'm still actively developing this project, so this section is a work-in-progress**
Anything with a question mark in the subheading is more like an idea I have to improve model performance rather than an existing model feature.

### Removal of Decoder Layers
This is the most significant change from the full encoder-decoder transformer.
For this prediction problem, I'm only looking to leverage the self-attention mechanism to enable the model to learn relationships between molecules in the polymer, which may be useful in predicting the target properties.
I don't need to predict the next character in a SMILES string and, in general, I don't need to convert the context-rich higher-dimensional representations of SMILES strings back to characters.
While it's possible to train an encoder-decoder transformer with SMILES data and then only use the encoder half, I hypothesized that it would be more effective to train the model to learn representations that are useful strictly in the context of predicting the relevant properties.
At a practical level, discarding the decoder parameters allows training on limited VRAM.
The encoder-only transformer outputs its learned representation of SMILES strings to a fully-connected feedforward network, which then outputs the five target predictions.
(Note that the FFN network mentioned here is not one of the FFN networks that occur in layers of transformers, it's instead an entirely separate FFN from the transformer).

### Masking
There is no casual masking of SMILES characters in this model (though I do still mask pad tokens).
While other encoder-only models like the BERT family mask random tokens and train to predict them, this model isn't meant to predict SMILES characters.
It's training to predict polymer properties directly, and the full context of all molecules in the polymer would be best for doing so.

### Relative Positional Encoding?
My materials science knowledge is limited and I'm not an expert in the SMILES convention.
My best guess is that the absolute position of a molecule in the SMILES isn't the most relevant property, but rather its relative position to other molecules is what defines a polymer's properties
(I imagine this relates more directly to physical structure than absolute position).
An implementation of relative positional encoding may help the model learn positional relationships more efficiently than the attention mechanism.

### CNN in between transformer and FFN?
Currently, the output of the transformer-encoder part of the model is flattened and fed directly into a feedforward network (FFN input is of dimension sequence_length x d_model).
This sudden reduction in dimensionality (~64x) may lead to the loss of information developed in the self-attention mechanism.
A convolutional architecture could help alleviate this by first compressing each character in the SMILES string to a single neuron output, which is then fed into a linear layer with an input dimension of sequence length.
This could also lead to its own form of information loss however, since entries in each character's self-attention-learned representation can't interact with entries in other character's representations directly. 

### Multivariate vs. univariate regression?
This could be a high-impact modification to the model.
Instead of predicting all five target values simultaneously, I could use five separate models that each predict one of the target values.
Each model could then optimize to predict one property better than a multivariate regression single model would, however, this may lead to issues with generalizability (i.e., predicting multiple properties forces the model to learn instead of memorize?).
