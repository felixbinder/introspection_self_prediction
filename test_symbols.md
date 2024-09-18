# Functional proposal III

Original Model: $M_A \in \mathcal{M}$, second model $M_B \in \mathcal{M}$

Self-prediction: $I_{M\in\mathcal{M}} : \mathcal{M} \to \mathcal{M}$, for the trained model $I_{M_A}(M_A)$

Cross-prediction: $I_{M_A}(M_B)$

Induced shift: $\text{shift}(I_{M_A}(M_A))$ (or maybe $\text{shift}_B(I_{M_A}(M_A))$)



# James'  proposal

Original Model: $M$, second model $A$

Self-prediction: $M_{M}$, $A_{A}$ # Subscript is the data used to train the model

Cross-prediction: $A_{M_{M}}$ # Train model A on the data of model $M_{M}$

Induced shift: $\text{shift}(M_{M})$
