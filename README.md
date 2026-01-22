# Fermi-LATV2-DL

The full pipeline starting from patches of sky to catalog-like data products are created viathe pipeline as below: 

)

![pipeline](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/ASID_flow_updated_S10_Flux1.pdf)


We have these components listed below for constructing a deep learning based gamma-ray catalog for Fermi-LAT data. 

## Point Source Detection + Localization
* We use a multi-input UNET (takes in array-inputs of different shapes) to predict reliable masks around source locations.
* We then find the source center with Laplacian of Gaussian.
* Eco-system is based on Python (v 3.8.8); Other libraries are listed below:
    - TensorFlow; 2.4.1
    - Pandas; 1.4.2
    - Numpy; 1.22.3
    - Astropy; 5.0.4
    - Scikit-Image; 0.16.2
    - Scikit-Learn; 1.0.2
    
    
An example of segmentation predictions for 3 randomly selected patches are shown below: 

![segment](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/test_random_preds_masks2-7GeV.png)      


## Source Characterization: Flux Estimation
* We use a simple VGG-like network to estimate photon flux of individual sources (all detection, True Positives + False Positives)
   - A regression network, that essentially learns convert photon counts to photon fluxes above 1 GeV.

## Source Characterization: Classification
* Similar VGG-like network to flux estimation, but predicts the binary class of individual sources.
* Binary classification of individual detected sources to be either True or Fake (background fluctuations).
* Probability is calibrated via AUC-ROC; Shifts from standard 0.50 threshold to 0.11.

## Source Characterization: Location Uncertainty
* VGG-like base network with multiple inputs (image arrays + predicted source coordinates) to predict refined location + uncertainties
* A deep-ensemble regression network that has multiple outputs (x, y, dx, dy);

### Systematic Uncertainty: Background Model Independence

### Catalog Creation: Application on Real Data
