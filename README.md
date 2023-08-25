# 3rd Place Solution of Kaggle Contrails Competition
This is the Preferred Contrail's solution for [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming).

## Dataset
Please prepare the competition dataset under `input/`.
The directory can also be changed by `--in_base_dir` argument of `train.py`.

## Usage
### Train
Train 2.5D resnest269e:
```
python train.py --config_path config/resnest-2-5d.yaml --save_model --out_base_dir result/resnest
```
By this, `model.pt` is saved under `result/resnest/-1/`.

You can use different config files under `config/` to train different models.

### Infer and make submission
Make submission by previously trained resnest269e:
```
python make_submission.py --model_dirs result/resnest/-1
```
By this, you can output `submission.csv`.
You can also ensemble multiple models' predictions by giving space-separated directories to `--model_dirs`.

### Predict psuedo label (optional)
Make psuedo labels by previously trained resnest269e:
```
python make_pl.py --model_dirs result/resnest/-1
```
By this script, you can generate pseudo label under `pseudo-label/` (which can be changed by `--out_dir` option) for frames 2-7.
You can also ensemble multiple models' predictions by giving space-separated directories to `--model_dirs`.
We uploaded the pseudo labels that we created by several models as a Kaggle dataset ([link](https://www.kaggle.com/datasets/knshnb/contrails-pseudo-label)).

By specifying `pseudo_dir` (defaults to `None`) to `psudo-label` in config files, you can use the pseudo labels to pretrain models by `train.py`.
After that, you can finetune the model on the original data by setting `pretrained_model_path` to the pretrained model's directory.

## Reproducing the score
The best single model achieved 0.706/0.71770/0.71629 (validation/private/public), which could still win 3rd place. I uploaded this model as a Kaggle dataset ([link](https://www.kaggle.com/datasets/knshnb/contrails-maxvitlarge-best)).
By ensembling 18 models with different backbones and settings (the best configurations are under `config/`), I achieved 0.72233 of private LB.

As a final submission, we achieved a score of 0.72305 by ensembling team members' models with the ratio of `knshnb:yiemon:charmq = 0.80:0.15:0.05`.

Models were trained mainly on 2x or 4x NVIDIA A100 (80GB).

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [3rd Place Solution: 2.5D U-Net](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430685) Kaggle discussion.
