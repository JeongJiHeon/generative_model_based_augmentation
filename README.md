# README

First, we trained a score-network ( We customized the code of [Original Link](https://github.com/yang-song/score_sde_pytorch))

## Phase 1 [Training generative model]

[ Training Code in score_sde_pytorch folder ]
<pre><code>python main.py --config=config/T1FLAIR.py --outdir='./' --mode='train'
</code></pre>

<br/>
<br/>
<br/>

After training mutant/wild model, synthesis data can be generated by models.

## Phase 2 [Generating synthesis data]

[ Training Code in classification folder ]
<pre><code>python generate.py
</code></pre>

<br/>
<br/>
<br/>

## Phase 3 [Training classification model]

Next, we trained a classification network using 'classifcation' folder (ResNet50)

[ Training Code in classification folder ]
<pre><code>python train.py --num=0 --fake_ratio=1 --backbone='resnet50'
</code></pre>
