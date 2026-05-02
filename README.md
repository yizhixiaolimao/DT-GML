# DT-SMF

## ✨ Overall Study Overview

<p align="center">
  <img src="img/overview1.png" alt="Model Structure" width="1000"/>
</p>

---

## 💡 Primary Contribution
- **🧬 Dynamic multimodal framework:** We propose **DT-SMF** for multimodal early Alzheimer's disease analysis using structural MRI and clinical scale information.

- **🩺 SMC-oriented diagnosis:** DT-SMF focuses on **subjective memory complaints (SMC)**, an early and clinically important stage that has received limited attention in previous AD studies.

- **🔄 Unified clinical pipeline:** DT-SMF jointly supports early cognitive state diagnosis and MCI-to-AD progression prediction within a unified framework.

- **🔍 Interpretable prediction:** Plasma biomarker correlation and brain-region visualization demonstrate the biological plausibility of model predictions.

- **💻 Open-source release:** We provide source code and implementation details to support reproducibility and further research.
<p align="center">
  <img src="img/exp.png" alt="Model Structure" width="1000"/>
</p>
---

## 🧠 Interpretability

<p align="center">
  <img src="img/brain.png" alt="Model interpretability visualization of key brain regions" width="1000"/>
</p>

<p align="center">
  <img src="img/linear.png" alt="Model interpretability visualization of correlations with plasma biomarkers" width="1000"/>
</p>

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
<h2 style="color:blue;">Overall Evaluation Metrics</h2>

<table>
  <tr>
    <th>Models</th>
    <th>Acc ↑</th>
    <th>F1 ↑</th>
    <th>Sen ↑</th>
    <th>Prec ↑</th>
    <th>MCC ↑</th>
    <th>Kappa ↑</th>
  </tr>

  <tr>
    <td>DAFT</td>
    <td>67.80 ± 13.84</td>
    <td>66.84 ± 12.55</td>
    <td>70.37 ± 5.73</td>
    <td>64.71 ± 17.41</td>
    <td>53.11 ± 17.46</td>
    <td>51.45 ± 19.94</td>
  </tr>

  <tr>
    <td>MMGLF</td>
    <td>73.76 ± 5.43</td>
    <td>70.98 ± 7.11</td>
    <td>61.62 ± 34.98</td>
    <td>69.12 ± 10.89</td>
    <td>61.32 ± 7.66</td>
    <td>59.30 ± 8.91</td>
  </tr>

  <tr>
    <td>PANIC</td>
    <td><u>79.44 ± 4.73</u></td>
    <td><u>76.84 ± 5.93</u></td>
    <td><u>81.10 ± 5.09</u></td>
    <td><u>77.88 ± 5.84</u></td>
    <td><u>69.75 ± 6.42</u></td>
    <td><u>68.12 ± 7.70</u></td>
  </tr>

  <tr>
    <td>MultimodalAD</td>
    <td>70.53 ± 10.07</td>
    <td>72.33 ± 8.09</td>
    <td>76.60 ± 6.06</td>
    <td>69.54 ± 10.65</td>
    <td>60.37 ± 10.44</td>
    <td>56.83 ± 13.72</td>
  </tr>

  <tr>
    <td>Hyperfusion</td>
    <td>71.37 ± 4.23</td>
    <td>71.19 ± 3.96</td>
    <td>71.74 ± 4.88</td>
    <td>71.25 ± 4.63</td>
    <td>57.70 ± 6.71</td>
    <td>56.93 ± 6.31</td>
  </tr>

  <tr>
    <td><b>Ours</b></td>
    <td><b>84.86 ± 2.76</b></td>
    <td><b>83.46 ± 3.29</b></td>
    <td><b>85.29 ± 2.00</b></td>
    <td><b>84.07 ± 3.52</b></td>
    <td><b>77.38 ± 4.12</b></td>
    <td><b>76.77 ± 4.24</b></td>
  </tr>
</table>

<p>
  The best results are shown in <b>bold</b>, and the second-best results are <u>underlined</u>.
</p>

<h2 style="color:blue;">Progression Prediction</h2>

<p>
  Comparison of DT-SMF against models on the progression prediction task.
  The best results are shown in <b>bold</b>, and the second-best results are <u>underlined</u>.
</p>

<table>
  <tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">MRI</th>
    <th rowspan="2">TAB</th>
    <th colspan="6" style="text-align:center;">Short term Evaluation Metrics</th>
  </tr>
  <tr>
    <th>Acc ↑</th>
    <th>F1 ↑</th>
    <th>Sen ↑</th>
    <th>Prec ↑</th>
    <th>MCC ↑</th>
    <th>Kappa ↑</th>
  </tr>

  <tr>
    <td>CNN-TLSTM</td>
    <td>✓</td>
    <td>—</td>
    <td>71.15</td>
    <td>64.94</td>
    <td>63.00</td>
    <td>67.00</td>
    <td>39.20</td>
    <td>38.10</td>
  </tr>

  <tr>
    <td>Multi-ERMHA</td>
    <td>✓</td>
    <td>—</td>
    <td>71.15</td>
    <td>60.17</td>
    <td>56.00</td>
    <td>65.00</td>
    <td>30.10</td>
    <td>28.70</td>
  </tr>

  <tr>
    <td>Ours</td>
    <td>✓</td>
    <td>—</td>
    <td><u>82.75</u></td>
    <td><u>78.47</u></td>
    <td><u>77.00</u></td>
    <td><u>80.00</u></td>
    <td><u>61.20</u></td>
    <td><u>59.80</u></td>
  </tr>

  <tr>
    <td><b>Ours</b></td>
    <td>✓</td>
    <td>✓</td>
    <td><b>94.12</b></td>
    <td><b>92.50</b></td>
    <td><b>93.00</b></td>
    <td><b>92.00</b></td>
    <td><b>88.90</b></td>
    <td><b>87.60</b></td>
  </tr>
</table>

## Citation

```

```
