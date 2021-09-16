# Graz<sup>+</sup> technique

We develop a regularization technique to train convolutional neural network (CNN) classifiers utilizing relevance-guided heat maps calculated online during training. The developed relevance-guided framework achieves higher classification accuracies than conventional CNNs but more importantly, it relies on less but more relevant and physiological plausible voxels within brain tissue. Additionally, preprocessing effects from skull stripping and registration are mitigated, rendering this practically useful in deep learning neuroimaging studies.

![Graz_plus_results_overview](https://user-images.githubusercontent.com/16046522/131127533-f4773551-749a-4254-9992-d1e2b8d6118d.png)
Mean heat maps (highest relevances in yellow, overlaid on MNI152 template) and balanced classification accuracy (percentage). Unmasked and masked CNN classifiers obtain relevant image features overwhelmingly from global volumetric information (left and center columns), whereas Graz+ exclusively relies on  deep gray and white matter tissue adjacent to the ventricles (right column).

The preprint paper describing the Graz<sup>+</sup> technique is available under [C Tinauer et al., Explainable Brain Disease Classification and Relevance-Guided Deep Learning, medRxiv: 2021.09.09.21263013, 2021](https://doi.org/10.1101/2021.09.09.21263013).

More infos about our work on explainable AI can be found under [http://www.neuroimaging.at/pages/research/explainable-ai.php](http://www.neuroimaging.at/pages/research/explainable-ai.php).
