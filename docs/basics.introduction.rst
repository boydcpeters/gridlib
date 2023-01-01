.. _basics.introduction:

Introduction
============

In the following sections, it is explained how to work with GRIDLib and some example
analyses are shown. However, if you decide to perform GRID analysis, it is highly
recommended to read the paper where the original authors present the ideas and the
mathematics behind GRID:

Reisser, M., Hettich, J., Kuhn, T., Popp, A.P., Große-Berkenbusch, A. and Gebhardt,
J.C.M. (2020). Inferring quantity and qualities of superimposed reaction rates from
single molecule survival time distributions. Scientific Reports, 10(1).
doi:10.1038/s41598-020-58634-y.


Glossary
--------

During the rest of the guide, some terms of the original paper are used. Here is a
glossary of some of those terms:

.. glossary::

    Survival time distribution
        A survival time distribution is a function that gives the probability that an
        object of interest will survive past a certain time.

    GRID
        Abbreviation for Genuine Rate IDentification.
    
    Integration time
        The time that the laser and camera are turned on during an acquisition.
        (:math:`\tau_{\mathrm{int}}`) 

    Dark time
        Time that the laser and camera are turned off between acquisitions.
        (:math:`\tau_{\mathrm{d}}`) 

    Time-lapse time
        The sum of the integration time and the dark time. (:math:`\tau_{\mathrm{tl}}`) 

    Photobleaching rate
        The rate at which photobleaching happens. (:math:`k_{\mathrm{b}}`) 
    
    Photobleaching number
        The photobleaching rate multiplied by the integration time.
        (:math:`a=k_{\mathrm{b}}\times\tau_{\mathrm{int}}`) 
