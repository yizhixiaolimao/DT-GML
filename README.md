# MAG-ADNet

## ✨ Model Structure

<p align="center">
  <img src="img/model.png" alt="Model Structure" width="1000"/>
</p>

---

## 💡 Primary Contribution
To further enhance diagnostic accuracy and strengthen the effectiveness of intermediate-to-late multimodal fusion, we propose MAG-ADNet (Modality-Adversarial-Gated Network for AD). This is a novel method which not only optimizes the fusion strategy but also achieves efficient extraction of discriminative information.Our key contributions are summarized as follows:
1. 🌊The MAG-ADNet architecture achieves accurate diagnosis with dual modalities by leveraging two modality-specific ResNet-50 encoders together with the MDIF module. This design adversarially learns both modality-specific and fused representations.
2. 🌊We introduce Gated Multi-Head Self Attention (GMHSA) to enhance information flow across attention heads and strengthen cross-modal interactions.
3. 🌊We adopt DyT to enhance representation learning with a learnable, smoothly adjustable activation curve, improving robustness to distribution shifts and preserving fine-grained multimodal cues.
4. 🌊Extensive experiments on both the AD vs CN and LMCI vs EMCI datasets demonstrate superior performance of MAG-ADNet, achieving state-of-the-art results on most metrics.
---

## 🧠 Interpretability

<p align="center">
  <img src="img/interpre.png" alt="Model interpretability visualization" width="1000"/>
</p>

<p align="center">
  <em>Example: Occlusion Sensitivity heatmap showing the key brain regions the model focuses on in MRI / PET.</em>
</p>

---

## 📂 Dataset Structure
We employed the baseline FDG-PET and T1-weighted MRI data, preprocessed and provided by the [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/)
 database.
The dataset should be organized in the following format:

```text
├── mri                             # MRI images
│   ├── train
│   │   ├── AD    # Alzheimer's Disease
│   │   ├── MCI   # Mild Cognitive Impairment
│   │   ├── EMCI  # Early MCI
│   │   └── LMCI  # Late MCI
│   └── test
│       ├── AD
│       ├── MCI
│       ├── EMCI
│       └── LMCI
│
├── pet                             # PET images
│   ├── train
│   │   ├── AD
│   │   ├── MCI
│   │   ├── EMCI
│   │   └── LMCI
│   └── test
│       ├── AD
│       ├── MCI
│       ├── EMCI
│       └── LMCI
│


- `mri/` : contains structural MRI scans in NIfTI format (`.nii.gz`).  
- `pet/` : contains corresponding PET scans in NIfTI format (`.nii.gz`).  
```
---

## 📦 Environment
- Python <3.12.4>
- PyTorch >= <2.4.0>

"Quickstart: Create an Environment (Example)"：
```bash
conda create -n <env_name> python=3.9 -y
conda activate <env_name>
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Train & Test

To train and evaluate the model, simply run:

```bash
python main.py
```

---

## 📊 Results
<h2 style="color:blue;">Comparative Experiment</h2>
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="6" style="text-align:center;">AD vs CN</th>
    <th colspan="6" style="text-align:center;">LMCI vs EMCI</th>
  </tr>
  <tr>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
  </tr>
  <tr>
    <td>Miccai-Fusion</td>
    <td>86.25</td><td>86.14</td><td>72.29</td><td>86.11</td><td>83.78</td><td>72.32</td>
    <td>78.06</td><td>76.41</td><td>53.36</td><td>61.54</td><td>81.63</td><td>54.70</td>
  </tr>
  <tr>
    <td>ADViT</td>
    <td>85.00</td><td>84.38</td><td>69.07</td><td>72.22</td><td>92.86</td><td>70.59</td>
    <td>75.48</td><td>74.46</td><td>49.00</td><td>66.15</td><td>72.88</td><td>49.16</td>
  </tr>
  <tr>
    <td>Transmf_AD</td>
    <td>77.50</td><td>76.98</td><td>54.08</td><td>69.44</td><td>78.12</td><td>54.36</td>
    <td>62.58</td><td>57.78</td><td>20.53</td><td>19.73</td><td>40.00</td><td>47.27</td>
  </tr>
  <tr>
    <td>MENet</td>
    <td>87.50</td><td>87.37</td><td>74.75</td><td>86.11</td><td>86.11</td><td>74.75</td>
    <td>76.13</td><td>73.82</td><td>48.68</td><td>55.38</td><td>81.82</td><td>50.89</td>
  </tr>
  <tr>
    <td>Diamond</td>
    <td>85.00</td><td>82.53</td><td>71.57</td><td>72.22</td><td><b>96.29</td><td>73.59</td>
    <td>75.72</td><td>71.42</td><td>51.89</td><td><b>69.23</td><td>73.77</td><td>51.97</td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>92.50</b></td><td><b>92.42</b></td><td><b>84.85</b></td><td><b>91.67</b></td><td>91.67</b></td><td><b>84.85</b></td>
    <td><b>81.29</b></td><td><b>80.09</b></td><td><b>60.48</b></td><td>67.69</b></td><td><b>84.61</b></td><td><b>61.46</b></td>
  </tr>
</table>

---

<h2 style="color:blue;">Ablation Experiment</h2>
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="6" style="text-align:center;">AD vs CN</th>
    <th colspan="6" style="text-align:center;">LMCI vs EMCI</th>
  </tr>
  <tr>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
    <th>Acc</th>
    <th>F1</th>
    <th>Kappa</th>
    <th>Sen</th>
    <th>Prec</th>
    <th>Mcc</th>
  </tr>
  <tr>
    <td>(w/o PET)
    </td><td>76.25</td><td>75.79</td><td>51.65</td><td>69.44</td><td>75.76</td><td>51.81</td>
    <td>77.42</td><td>75.44</td><td>51.67</td><td>58.46</td><td>82.61</td><td>53.55</td>
  </tr>
  <tr>
    <td>(w/o MRI)</td>
    <td>86.25</td><td>85.99</td><td>72.01</td><td>80.56</td><td>87.88</td><td>72.22</td>
    <td>78.71</td><td>77.50</td><td>55.23</td><td>66.15</td><td>79.63</td><td>55.85</td>
  </tr>
  <tr>
    <td>(w/o MDIF)</td>
    <td>70.00</td><td>69.70</td><td>39.39</td><td>66.67</td><td>66.67</td><td>39.39</td>
    <td>68.39</td><td>62.81</td><td>29.84</td><td>35.38</td><td>76.67</td><td>34.48</td>
  </tr>
  <tr>
    <td>(w/o DyT)</td>
    <td>85.00</td><td>84.38</td><td>69.07</td><td>72.22</td><td><b>92.86</td><td>70.59</td>
    <td>77.42</td><td>76.13</td><td>52.52</td><td>64.62</td><td>77.78</td><td>53.11</td>
  </tr>
  <tr>
    <td>(w/o GMHSA)</td>
    <td>90.00</td><td>89.84</td><td>79.70</td><td>86.11</td><td>91.18</td><td>79.80</td>
    <td>80.65</td><td>78.69</td><td>58.30</td><td>60.00</td><td><b>90.70</td><td>61.23</td>
  </tr>
  <tr>
    <td><b>Ours</b></td>
    <td><b>92.50</b></td><td><b>92.42</b></td><td><b>84.85</b></td><td><b>91.67</b></td><td>91.67</b></td><td><b>84.85</b></td>
    <td><b>81.29</b></td><td><b>80.09</b></td><td><b>60.48</b></td><td><b>67.69</b></td><td>84.61</b></td><td><b>61.46</b></td>
  </tr>
</table>

## Citation

```

```
