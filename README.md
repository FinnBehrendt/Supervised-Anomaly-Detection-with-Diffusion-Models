# Supervised-Anomaly-Detection-with-Diffusion-Models

Codebase for our MIDL2024 Paper [Combining Reconstruction-based Unsupervised Anomaly Detection with Supervised Segmentation for Brain MRIs](https://openreview.net/forum?id=iWfUcg4FrD&noteId=iWfUcg4FrD).


## Abstract
In contrast to supervised deep learning approaches, unsupervised anomaly detection (UAD) methods can be trained with healthy data only and do not require pixel-level annotations, enabling the identification of unseen pathologies. While this is promising for clinical screening tasks, reconstruction-based UAD methods fall short in segmentation accuracy compared to supervised models. Therefore, self-supervised UAD approaches have been proposed to improve segmentation accuracy. Typically, synthetic anomalies are used to train a segmentation network in a supervised fashion. However, this approach does not effectively generalize to real pathologies.
We propose a framework combining reconstruction-based and self-supervised UAD methods to improve both segmentation performance for known anomalies and generalization to unknown pathologies. The framework includes an unsupervised diffusion model trained on healthy data to produce pseudo-healthy reconstructions and a supervised Unet trained to delineate anomalies from deviations between input-reconstruction pairs. 
Besides the effective use of synthetic training data, this framework allows for weakly-supervised training with small annotated data sets, generalizing to unseen pathologies. Our results show that with our approach, utilizing annotated data sets during training can substantially improve the segmentation performance for in-domain data while maintaining the generalizability of reconstruction-based approaches to pathologies unseen during training.

## Code

The code will be available soon.
## Citation
    @inproceedings{behrendt2024combining,
      title={Combining Reconstruction-based Unsupervised Anomaly Detection with Supervised Segmentation for Brain MRIs},
      author={Behrendt, Finn and Bhattacharya, Debayan and Maack, Lennart and Kr{\"u}ger, Julia and Opfer, Roland and Schlaefer, Alexander},
      booktitle={Medical Imaging with Deep Learning},
      year={2024},
      url={https://openreview.net/forum?id=iWfUcg4FrD},
    }


