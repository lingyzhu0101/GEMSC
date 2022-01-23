# GEMSC
Enlightening Low-Light Images with DynamicGuidance for Context Enrichment

Images acquired in low-light conditions suffer from a series of visual quality degradations, \textit{e.g.}, low visibility, degraded contrast, and intensive noise. These complicated degradations based on various contexts  (\textit{e.g}., noise in smooth regions, over-exposure in well-exposed regions and {low contrast around edges}) cast major challenges to the low-light image enhancement. Herein, we propose a new methodology by imposing a learnable guidance map from the signal and deep priors, making the deep neural network  adaptively enhance low-light images in a region-dependent manner. The enhancement capability of the learnable guidance map is further exploited with the multi-scale dilated context collaboration, leading to contextually enriched feature representations extracted by the model with various receptive fields. Through assimilating the intrinsic perceptual information from the learned guidance map, richer and more realistic textures are generated.
Extensive experiments on real low-light images demonstrate the effectiveness of our method, which delivers superior results quantitatively and qualitatively.

# env requirement
pip install torch==0.4.0
