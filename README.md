# Bayesian estimation of the water proportion on Earth

The estimation is done by tossing a globe several times and recording if the point of contact (your finger) is on Water or Land.

This sequence of W, L observations is used to estimate a *posterior* probability density on the possible values of the proportion $p$, by assuming a uninformative *prior* distribution (a "flat" prior).

Does this flat prior correspond to your knowledge before the experiment? How could you code your knowledge in your own prior? 

A good reference on this kind of reasoning is: [R. McElreath "Statistical Rethinking"](https://xcelab.net/rm/) (The book is R based, but there are many independent translations to Python e PyMC, check the book website).
