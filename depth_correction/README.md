## Depth Correction
1. [Description](#description)
1. [DRIT](#drit)
1. [Example](#example)
### Description
Correct predicted depth according to the ground truth ordinal depth. Ordinal depth is expected to be encoded as $10 \times intradepth + interdepth$, and unannotated parts are mapped to pad_token (*default: 209*).

Depth correction algorithms: simple, shift, and affine. Correction is done for each image separately, and so for each [inter/intra]depth. "Global" refers to image context; "current" refers to values on particular [inter/intra]depth plane.
* `simple` - clipping pixel values to the interval $[min_{global}; max_{global}]$ without any modification of the values;
* `shift` - same as `simple`, but before clipping values are shifted so that the minimum value equals $min_{global}$;
* `affine` - a linear transformation of values that ensures: $min_{current} \leftarrow min_{global},\ max_{current} \leftarrow max_{global}$.
### DRIT
```bash
git clone https://github.com/HsinYingLee/DRIT.git
```
To apply style transfer neural network one can either train DRIT from scratch (`train_drit.py`) or download a pretrained on dcm and COCO model.
* For training replace paths to data if needed (*default: dcm-dataset/panels and val2017*), and run
  ```bash
  python train_drit.py
  ```
* Download a pretrained model
  ```bash
  gdown https://drive.google.com/file/d/1PIagZhuKsassqtAgeGixoOSynJwyg2ip
  ```
* Evaluation
  ```python
  sys.path.append('DRIT/src')
  
  args = Munch()
  # init args
  model = DRIT(args)
  model.to(device)
  model.resume('drit_weights.pth', train=False)
  model.eval()

  with torch.no_grad():
    res = model.test_forward(img) # transfer without a reference
    res = model.test_forward_transfer(img, img2) # img2 - a reference image
  ```

### Example
```python
from depth_correction import *

pred_depth = Image.open('depth.png').convert("L") # depth predicted by any neural network
gt_depth = Image.open('gt_depth.png').convert("L") # fround truth ordinal depth
corrected_depth = correct_depth(np.array(pred_depth), np.array(gt_depth), method=affine_correction)
sanity_check(np.array(gt_depth), corrected_depth) # check correction
```
