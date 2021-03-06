# pymln
Python implementation of unsupervised semantic parsing and markov logic network knowledgebase induction. This work is funded through DARPA’s <a href='https://www.darpa.mil/program/automating-scientific-knowledge-extraction'>ASKE</a> program (Automating Scientific Knowledge Extraction) as part of Gallup's <a href='https://github.com/GallupGovt/multivac'>MULTIVAC</a> project. This is a work in progress. 

## Overview
The `pymln` subsystem reads in the `*.dep`, `*.morph`, and `*.input` files generated by the prior parsing step, and compiles these into nested trees of nodes representing tokens and their parent-child relationships. These nodes are assigned to initial semantic clusters which link them via relationships and arguments to other nodes in the knowledge base. Each cluster tracks these links in python dictionaries and sets mapping types of relationships, types of arguments, and specific Argument nodes to their globally tracked indices.

## Markov Logic Networks
The results of this process are a graphical model of nodes and edges governed by first-order logic formulas which embed all the statements, entities and relationships found in the source data. These first-order logic formulas are assigned weights according to the frequency of their occurrence in the corpus and placed in a Markov network structure to create a MLN.

A Markov network is a set of random variables having a Markov property (where the conditional probability of future states is dependent solely on the present state) described by an undirected graph. In a MLN, the nodes of the network graph are atomic first-order formulas (atoms), and the edges are the logical connectives (here, dependencies) used to construct a larger formula. 

First-order logic (FOL), also known as first-order predicate calculus or first-order functional calculus, is a system in which each sentence, or statement, is broken down into a subject and a predicate. The predicate modifies or defines the properties of the subject. This system naturally mirrors the dependency tree parsing performed in the previous step.

Each formula is considered to be a clique (a subset of nodes in the graph such that every pair of nodes in the clique are connected), and the Markov blanket (the set of other nodes containing all the information necessary to determine the value of a given node) is the set of larger formulas in which a given atom appears. A “grounded atom” is an atomic formula with actual constants/values supplied to give the formula a “grounded” meaning. MLNs associate a weight with each formula, designated by the frequency with which that formula is “true” given its groundings in the available evidence (such as our corpus). Unlike in first-order logic knowledge bases, in a MLN when one clique or formula is violated (e.g., “Senators from Kansas are Republican”) the “world” described by that grounding is simply less probable, rather than impossible.<sup>[1](#1)</sup>

To finalize the domain’s model ontology, MULTIVAC clusters together semantically interchangeable formulas into more generalized versions, identified as formulas which can be combined to improve the log-likelihood of observing the given set of formulas, determined by the sum of the weights. For example, in the sentence “Glucocorticoid resistance in the squirrel monkey is associated with overexpression of the immunophilin fkbp51,” the formula for the component “resistance is associated with overexpression” (passive voice):


<p align='center'> <i> &#955;x1&#955;x2.associated(n1)/\agent(n1,“overexpression)/\nsubjpass(n1,resistance”) </i> </p>

is semantically the same as for “overexpression associates with resistance” (active voice):

<p align='center'> <i> &#955;x1&#955;x2.associated(n1)/\nsubj(n1,“overexpression)/\dobj(n1,resistance”) </i> </p>

and the clusters representing these formulas would then be merged into one cluster for the concept “&#955;x1&#955;x2.associated" linked to the (overexpression) and dependent (resistance) Argument nodes and the Relationship type (submit) nodes, that abstracts away the active/passive voice distinction in the statements.  

MMULTIVAC extends the existing MLN ontology concept by integrating the parsed model formulas along with the actual text, mapping both into the same shared ontological space. Thus, the dependencies and relationships in the models, as represented in the mathematical formulas associated with them, are also represented in the MLN ontology and enriched by the resulting relationships with the organic contextual knowledge provided by the natural language text. 

Another important element of our integration is the co-occurrence of formula parameters in the text of the containing article, where authors explain their models, terminology and approach. This allows our domain-adapted GloVe embedding model to assign word vectors to model parameters that relate them to other natural language words in the corpus. Calculations of semantic similarity can then be weighted by comparison (typically cosine similarity metrics) of the tokens involved when considering whether or which clusters to merge during the MLN construction.

The result is a domain ontology represented as a Markov Logic Network grounded on the models found in our domain’s scientific literature. The MLN represents a meta-model ontology architecture that can be queried not just for facts but for cause and effect inference, counter-factual explorations and uncertainty quantification across the domain.

### End Notes
<sup><a name='1'>1</a></sup> https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf <br>

<hr>

### This software is derived from the USP (Beta Version) Software by the University of Washington, available here: http://alchemy.cs.washington.edu/usp/ 



All of the documentation and software included in the USP (Beta Version) Software is copyrighted by Hoifung Poon and Pedro Domingos.


Copyright [2009-11] Hoifung Poon and Pedro Domingos. All rights reserved.


Contact: Hoifung Poon (hoifung.poon@gmail.com).


Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:


 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.


 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.


 3. All advertising materials mentioning features or use of this software must display the following acknowledgment: "This product includes software developed by Hoifung Poon and Pedro Domingos in the Department of Computer Science and Engineering at the University of Washington".


 4. Your publications acknowledge the use or contribution made by the Software to your research using the following citation(s): 

   Hoifung Poon and Pedro Domingos (2009). "Unsupervised Semantic Parsing", in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2009. http://alchemy.cs.washington.edu/usp.


 5. Neither the name of the University of Washington nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

