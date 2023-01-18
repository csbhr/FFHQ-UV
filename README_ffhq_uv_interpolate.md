## FFHQ-UV-Interpolate

**FFHQ-UV-Interpolate** is a variant of FFHQ-UV. It is based on latent space interpolation, which is with compromised diversity but higher quality and larger scale (**100,000** UV-maps).

We adopt the following main steps to obtain FFHQ-UV-Interpolate from FFHQ-UV:
- Automatic data filtering considering BS Error, valid texture area ratio, expression detection, etc.
- Sample classification considering attributes such as gender, age, beard, etc.
- Latent space interpolation within each sample category.

Some quantitative comparisons between FFHQ-UV and FFHQ-UV-Interpolate (the values of ID std. are divided by the value of FFHQ):  
|  Dataset   | ID std. $\uparrow$ | # UV-maps $\uparrow$ | BS Error $\downarrow$ |
|  ----  | ----  | ----  | ----  |
| FFHQ-UV  | 90.06% | 54,165 | 7.293 |
| FFHQ-UV-Interpolate  | 80.12% | 100,000 | 4.490 |
