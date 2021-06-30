# Orbit-Segmentation

In order to run the code, you have to download the 'pretrained model' and 'Orbit Dataset' and give the path as arg. For example:

```
python main.py --pretrained_model_path [PATH/TO/MODEL] --dataset_path [PATH/TO/DATASET]
```

For the other args, refer to the bash scripts.

In order to download the pretrained model, copy and paste the following command in your terminal:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Tw3vdebds4bkmhWLss8RMu8Xri1nhDok' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Tw3vdebds4bkmhWLss8RMu8Xri1nhDok" -O pretrained_model.zip && rm -rf /tmp/cookies.txt
```

In order to download the dataset used for the experiments, copy and paste the following command in your terminal:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zMEun8jLEUHseho3S_HWsmbZuBZVfuvN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zMEun8jLEUHseho3S_HWsmbZuBZVfuvN" -O data.zip && rm -rf /tmp/cookies.txt
```
