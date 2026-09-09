# Fermi-LATV2-DL

The full pipeline starting from patches of sky to catalog-like data products are created viathe pipeline as below:

![pipeline](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/asid_flow_S20.png)


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

![effective-recall](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/B1-B2-real-assoc-FGL-FL8Y_pre.png)

### Catalog Creation: Application on Real Data

#### **_1FDL catalog preview_**

First few entries from the full list of 3821 sources; Values are rounded for display;

| 1FDL source | Latitude _b_ (deg) | Longitude _l_ (deg) | 1FDL_Flux1000 (ph cm⁻² s⁻¹) | 68% conf-radius (deg;) | DR2 counterpart | DR2 ASSOC1 |
|---|---:|---:|---:|---:|---|---|
| 1FDL J1923.0+1409 | -0.415 | 49.097 | 4.138e-08 | 0.0278 | 4FGL J1923.2+1408e | W 51C |
| 1FDL J0616.8+2233 | 2.941 | 189.019 | 1.294e-07 | 0.0405 | 4FGL J0617.2+2234e | IC 443 |
| 1FDL J1635.9-4731 | -0.085 | 337.035 | 2.230e-08 | 0.0396 | 4FGL J1636.3-4731e | SNR G337.0-00.1 |
| 1FDL J1800.6-2352 | -0.336 | 6.066 | 6.225e-09 | 0.0966 | 4FGL J1759.7-2354 | — |
| 1FDL J1856.3+0114 | -0.531 | 34.603 | 6.057e-08 | 0.0767 | 4FGL J1855.9+0121e | W 44 |
| 1FDL J1801.5-2322 | -0.265 | 6.600 | 4.366e-08 | 0.0540 | 4FGL J1801.3-2326e | W 28 |
| 1FDL J1843.5-0347 | 0.026 | 28.648 | 1.523e-08 | 0.0766 | 4FGL J1844.4-0345 | PSR J1844-0346 |
| 1FDL J1855.6+0141 | -0.062 | 34.912 | 7.596e-09 | 0.0941 | 4FGL J1854.7+0153 | — |

![full-sky-map](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/Plot9_AllSky1FDL_DR2Class_V2.png)

![cat-overview](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/1FDL_catalog_overview.png)

#### Unassociated Bright DR2s (Signif\_Avg > 20)

Most of the bright unassociated DR2 sources are close to the plane and our algorithm either fails to localize them properly or fails to detect them in high-density regions; 
An example is shown in the Figure below:

![unassoc-bright-DR2](https://github.com/suvoooo/Fermi-LATV2-DL/blob/main/Images/check_UNet_pred_mask_2patches_patch_394_F_Mode_wExt.png)  
